import torch


class SpeculativeDecoder:

    def __init__(self, draft_model, target_model, draft_steps=4):

        self.draft = draft_model
        self.target = target_model
        self.steps = draft_steps

    @torch.no_grad()
    def generate(self, tokens):

        generated = tokens

        while generated.size(1) < 128:

            # draft model proposes tokens
            draft_tokens = []

            cur = generated

            for _ in range(self.steps):

                logits = self.draft(cur)

                next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)

                draft_tokens.append(next_token)

                cur = torch.cat([cur, next_token], dim=1)

            draft_tokens = torch.cat(draft_tokens, dim=1)

            # verify with large model
            full = torch.cat([generated, draft_tokens], dim=1)

            logits = self.target(full)

            probs = torch.softmax(logits[:, -draft_tokens.size(1):], dim=-1)

            accepted = []

            for i in range(draft_tokens.size(1)):

                token = draft_tokens[:, i]

                prob = probs[:, i, token]

                if torch.rand(1) < prob:

                    accepted.append(token)

                else:
                    break

            if len(accepted) > 0:

                accepted = torch.stack(accepted, dim=1)

                generated = torch.cat([generated, accepted], dim=1)

            else:

                logits = self.target(generated)

                token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)

                generated = torch.cat([generated, token], dim=1)

        return generated