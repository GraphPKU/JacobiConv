import torch


def train(optimizer, model, ds, loss_fn):
    optimizer.zero_grad()
    model.train()
    pred = model(ds.edge_index, ds.edge_attr, ds.pos)
    loss = loss_fn(pred, ds.y)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, ds, metrics, loss_fn=None):
    model.eval()
    pred = model(ds.edge_index, ds.edge_attr, ds.pos)
    y = ds.y
    loss = loss_fn(pred, y)
    return metrics(pred.cpu().numpy(), y.cpu().numpy()), loss



