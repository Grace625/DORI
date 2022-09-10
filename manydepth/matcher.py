import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from detectron2.structures.instances import Instances

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

class HungarianMatcher(nn.Module):

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, ins_threshold: float = 0.5):
        """
        Args:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        self.ins_threshold = ins_threshold

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"


    @torch.no_grad()
    def memory_efficient_forward(self, instances_n, instances_m, instances_0):
        """More memory-friendly matching"""
        
        N_n = len(instances_n)
        N_m = len(instances_m)
        N_0 = len(instances_0)

        pred_class_n = instances_n.pred_classes
        pred_class_m = instances_m.pred_classes
        pred_class_0 = instances_0.pred_classes


        class_n = pred_class_n.unsqueeze(1).repeat(1, N_0)
        class_m = pred_class_m.unsqueeze(1).repeat(1, N_0)
        class_0_1 = pred_class_0.repeat(N_n, 1)
        class_0_2 = pred_class_0.repeat(N_m, 1)

        cost_class1 = torch.where(class_n == class_0_1, 0, 1)
        cost_class2 = torch.where(class_m == class_0_2, 0, 1)

        masks_n = instances_n.pred_masks # shape = N_n, H, W
        masks_m = instances_m.pred_masks 
        masks_0 = instances_0.pred_masks # shape = N_0, H, W

        with autocast(enabled=False):
            masks_n = masks_n.flatten(1).float()
            masks_m = masks_m.flatten(1).float()
            masks_0 = masks_0.flatten(1).float()

            # Compute the dice loss betwen masks
            cost_dice1 = batch_dice_loss_jit(masks_n, masks_0)
            cost_dice2 = batch_dice_loss_jit(masks_m, masks_0)

        # Final cost matrix
        C1 = (
            self.cost_class * cost_class1
            + self.cost_dice * cost_dice1
        )
        C2 = (
            self.cost_class * cost_class2
            + self.cost_dice * cost_dice2
        )
       
        C1 = C1.cpu()
        C2 = C2.cpu()

        idx_n, idx_0 = linear_sum_assignment(C1)
        idx_m, idx_1 = linear_sum_assignment(C2)

        id_0 = [0] * N_0
        for i in range(len(idx_0)):
            id_0[idx_0[i]] = i

        id_1 = [0] * N_0
        for i in range(len(idx_1)):
            id_1[idx_1[i]] = i
        
        idx_intersection = set(idx_0) & set(idx_1)

        res0 = []
        res1 = []
        for idx in idx_intersection:
            ix0 = id_0[idx]
            ix1 = id_1[idx]
            res0.append(idx_n[ix0])
            res1.append(idx_m[ix1])
        
        slice_n = torch.as_tensor(res0, dtype=torch.long, device=pred_class_n.device)
        slice_m = torch.as_tensor(res1, dtype=torch.long, device=pred_class_n.device)

        return slice_n, slice_m


    @torch.no_grad()
    def forward(self, instances_n, instances_m, instances_0):
        """Performs the matching

        Args:
            instances_n: List of detectron2 Instances object, instances in I_{t-1->t}
            instances_m: List of detectron2 Instances object, instances in I_{t+1->t}
            instances_0: List of detectron2 Instances object, instances in I_{t}

        Returns:
            Instance ID correspondence for instances_n and instances_m.
            slice_n : ID of matched instances in instances_n (matched instances in I_{t-1->t})
            slice_m : ID of matched instances in instances_m (matched instances in I_{t-1->t})

        """
        return self.memory_efficient_forward(instances_n, instances_m, instances_0)
