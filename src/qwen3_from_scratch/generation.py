import torch


def generate(model, tokenizer, prompt, max_new_tokens):
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    model.eval()

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            sequence_length = input_ids.shape[1]
            position_ids = torch.arange(sequence_length, device=device).unsqueeze(0)
            mask = torch.full(
                (sequence_length, sequence_length),
                float("-inf"),
                device=device,
            )
            mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

            logits = model(input_ids, mask, position_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)
