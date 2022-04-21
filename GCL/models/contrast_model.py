import torch

from typing import Optional, Union

from GCL.losses import Loss
from GCL.models import get_dense_sampler
from GCL.models.sampler import DenseSampler, DefaultSampler, ContrastInstance


class SingleBranchContrast(torch.nn.Module):
    def __init__(self, loss: Loss, mode: str, intraview_negs: bool = False, **kwargs):
        super(SingleBranchContrast, self).__init__()
        assert mode == 'G2L'  # only global-local pairs allowed in single-branch contrastive learning
        self.loss = loss
        self.mode = mode
        self.sampler = get_dense_sampler(mode, intraview_negs=intraview_negs)
        self.kwargs = kwargs

    def forward(self, h, g, batch=None, hn=None, extra_pos_mask=None, extra_neg_mask=None):
        if batch is None:  # for single-graph datasets
            assert hn is not None
            ci = self.sampler(anchor=g, sample=h, neg_sample=hn)
        else:  # for multi-graph datasets
            assert batch is not None
            ci = self.sampler(anchor=g, sample=h, batch=batch)

        loss = self.loss(contrast_instance=ci, **self.kwargs)
        return loss


class DualBranchContrast(torch.nn.Module):
    def __init__(
            self, loss: Loss, mode: str, intraview_negs: bool = False,
            sampler: Optional[Union[DenseSampler, DefaultSampler]] = None, **kwargs):
        super(DualBranchContrast, self).__init__()
        self.loss = loss
        self.mode = mode
        if sampler is not None:
            self.sampler = sampler
        else:
            if (mode == 'L2L' or mode == 'G2G') and (not intraview_negs):
                self.sampler = DefaultSampler()
            else:
                self.sampler = get_dense_sampler(mode, intraview_negs=intraview_negs)
        self.kwargs = kwargs

    def forward(
            self, h1=None, h2=None, g1=None, g2=None, batch=None, h3=None, h4=None,
            extra_pos_mask=None, extra_neg_mask=None,
            pos_mask1=None, pos_mask2=None, neg_mask1=None, neg_mask2=None):

        if isinstance(self.sampler, DefaultSampler) \
                and not (extra_neg_mask is None and extra_pos_mask is None):  # sanity check
            raise RuntimeError('Default sampler does not support additional positive and negative samples')

        if self.mode == 'L2L':
            assert h1 is not None and h2 is not None
            ci1 = self.sampler(
                anchor=h1, sample=h2, pos_mask=pos_mask1, neg_mask=neg_mask1,
                extra_pos_mask=extra_pos_mask, extra_neg_mask=extra_neg_mask)
            ci2 = self.sampler(
                anchor=h2, sample=h1, pos_mask=pos_mask2, neg_mask=neg_mask2,
                extra_pos_mask=extra_pos_mask, extra_neg_mask=extra_neg_mask)
        elif self.mode == 'G2G':
            assert g1 is not None and g2 is not None
            ci1 = self.sampler(
                anchor=g1, sample=g2, pos_mask=pos_mask1, neg_mask=neg_mask1,
                extra_pos_mask=extra_pos_mask, extra_neg_mask=extra_neg_mask)
            ci2 = self.sampler(
                anchor=g2, sample=g1, pos_mask=pos_mask2, neg_mask=neg_mask2,
                extra_pos_mask=extra_pos_mask, extra_neg_mask=extra_neg_mask)
        else:  # global-to-local
            if batch is None or batch.max().item() + 1 <= 1:  # single graph
                assert all(v is not None for v in [h1, h2, g1, g2, h3, h4])
                ci1 = self.sampler(
                    anchor=g1, sample=h2, neg_sample=h4,
                    extra_pos_mask=extra_pos_mask, extra_neg_mask=extra_neg_mask)
                ci2 = self.sampler(
                    anchor=g2, sample=h1, neg_sample=h3,
                    extra_pos_mask=extra_pos_mask, extra_neg_mask=extra_neg_mask)
            else:  # multiple graphs
                assert all(v is not None for v in [h1, h2, g1, g2, batch])
                ci1 = self.sampler(
                    anchor=g1, sample=h2, batch=batch,
                    extra_pos_mask=extra_pos_mask, extra_neg_mask=extra_neg_mask)
                ci2 = self.sampler(
                    anchor=g2, sample=h1, batch=batch,
                    extra_pos_mask=extra_pos_mask, extra_neg_mask=extra_neg_mask)

        l1 = self.loss(contrast_instance=ci1, **self.kwargs)
        l2 = self.loss(contrast_instance=ci2, **self.kwargs)

        return (l1 + l2) * 0.5


class BootstrapContrast(torch.nn.Module):
    def __init__(self, loss, mode='L2L', sampler: Optional[Union[DenseSampler, DefaultSampler]] = None):
        super(BootstrapContrast, self).__init__()
        self.loss = loss
        self.mode = mode
        if sampler is not None:
            self.sampler = sampler
        else:
            if mode == 'L2L' or mode == 'G2G':
                self.sampler = DefaultSampler()
            else:
                self.sampler = get_dense_sampler(mode, intraview_negs=False)

    def forward(
            self, h1_pred=None, h2_pred=None, h1_target=None, h2_target=None,
            g1_pred=None, g2_pred=None, g1_target=None, g2_target=None,
            batch=None, extra_pos_mask=None):
        if self.mode == 'L2L':
            assert all(v is not None for v in [h1_pred, h2_pred, h1_target, h2_target])
            ci1 = self.sampler(anchor=h1_target, sample=h2_pred)
            ci2 = self.sampler(anchor=h2_target, sample=h1_pred)
        elif self.mode == 'G2G':
            assert all(v is not None for v in [g1_pred, g2_pred, g1_target, g2_target])
            ci1 = self.sampler(anchor=g1_target, sample=g2_pred)
            ci2 = self.sampler(anchor=g2_target, sample=g1_pred)
        else:
            assert all(v is not None for v in [h1_pred, h2_pred, g1_target, g2_target])
            if batch is None or batch.max().item() + 1 <= 1:  # single graph
                # pos_mask1 = pos_mask2 = torch.ones([1, h1_pred.shape[0]], device=h1_pred.device)
                ci1 = ContrastInstance(anchor=g1_target, sample=h2_pred)
                ci2 = ContrastInstance(anchor=g2_target, sample=h1_pred)
            else:
                ci1 = self.sampler(anchor=g1_target, sample=h2_pred, batch=batch)
                ci2 = self.sampler(anchor=g2_target, sample=h1_pred, batch=batch)

        l1 = self.loss(contrast_instance=ci1)
        l2 = self.loss(contrast_instance=ci2)

        return (l1 + l2) * 0.5


class WithinEmbedContrast(torch.nn.Module):
    def __init__(self, loss: Loss):
        super(WithinEmbedContrast, self).__init__()
        self.loss = loss

    def forward(self, h1, h2):
        l1 = self.loss(contrast_instance=ContrastInstance(anchor=h1, sample=h2))
        l2 = self.loss(contrast_instance=ContrastInstance(anchor=h2, sample=h1))
        return (l1 + l2) * 0.5
