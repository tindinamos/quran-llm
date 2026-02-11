import torch


class Trainer:
    def __init__(self, model, device, eval_iters, block_size, batch_size):
        self.model = model
        self.device = device
        self.eval_iters = eval_iters
        self.block_size = block_size
        self.batch_size = batch_size

    def get_batch(self, data, split):
        # generate a small batch of data of inputs x and targets y
        dataset = data[split]
        ix = torch.randint(len(dataset) - self.block_size, (self.batch_size,))
        x = torch.stack([dataset[i : i + self.block_size] for i in ix])
        y = torch.stack([dataset[i + 1 : i + self.block_size + 1] for i in ix])
        return x, y

    @torch.no_grad()
    def estimate_loss(self, data):
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                x, y = self.get_batch(data, split)
                x = x.to(self.device)
                y = y.to(self.device)
                logits, loss = self.model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def run(self, data, max_iters, eval_interval, learning_rate):
        # create a PyTorch optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        for iteration in range(max_iters):
            # every once in a while evaluate the loss on train and val sets
            if iteration % eval_interval == 0:
                losses = self.estimate_loss(data)
                print(
                    f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )

            # sample a batch of data
            xb, yb = self.get_batch(data, "train")
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            # evaluate the loss
            logits, loss = self.model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
