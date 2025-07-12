# import numpy as np
# from tqdm import tqdm
# import torch
# from data import get_test_data


# @torch.no_grad()
# def eval(
#     model,
#     tokenizer,
#     dataset="wikitext2",
#     model_seq_len=2048,
#     batch_size=32,
#     device="cuda",
# ):
#     model.to(device)
#     model.eval()
#     test_loader = get_test_data(
#         dataset, tokenizer, seq_len=model_seq_len, batch_size=batch_size
#     )
#     nlls = []
#     for batch in tqdm(test_loader):
#         batch = batch.to(device)
#         output = model(batch, use_cache=False)
#         lm_logits = output.logits
#         if torch.isfinite(lm_logits).all():
#             shift_logits = lm_logits[:, :-1, :].contiguous()
#             shift_labels = batch[:, 1:].contiguous()
#             loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
#             loss = loss_fct(
#                 shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1)
#             )
#             nlls.append(loss)

#     perplexity = np.exp(torch.cat(nlls, dim=-1).mean().item())

#     return perplexity


import numpy as np
import torch
from tqdm import tqdm
from data import get_test_data


@torch.no_grad()
def eval(
    model,
    tokenizer,
    dataset="wikitext2",
    model_seq_len=2048,
    batch_size=32,
    device="cuda",
):
    model.to(device)
    model.eval()

    dataloader = get_test_data(
        dataset, tokenizer, seq_len=model_seq_len, batch_size=batch_size
    )
    nll_losses = []

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    for batch in tqdm(dataloader, desc=f"Evaluating on {dataset}"):
        batch = batch.to(device)
        outputs = model(batch, use_cache=False)
        logits = outputs.logits

        if not torch.isfinite(logits).all():
            continue

        # Shift for next-token prediction
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        nll_losses.append(loss)

    avg_nll = torch.cat(nll_losses).mean().item()
    perplexity = np.exp(avg_nll)

    return perplexity
