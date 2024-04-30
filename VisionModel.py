import torch.nn as nn
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoProcessor,
    AutoTokenizer
)
from MLP import ProjectionModule

class VisionModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.encoder_name = "google/siglip-so400m-patch14-384"
        self.encoder = AutoModel.from_pretrained(self.encoder_name)
        self.processor = AutoProcessor.from_pretrained(self.encoder_name)

        self.projector = ProjectionModule(1152, 3072)

        self.decoder_name = "microsoft/phi-3-mini-4k-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.decoder_name)
        self.decoder = AutoModelForCausalLM.from_pretrained(self.encoder_name)
        self.embedding_layer = self.decoder.get_input_embeddings()

        # disable grad for encoder and decoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        
    def forward(self, img, query, answer, targets=None):
        # encode the input
        img_processed = self.processor(img, return_tensors="pt")
        img_features = self.encoder.get_image_features(**img_processed) # a (1, 1152) tensor

        lm_img_features = self.projector(img_features) # a (1, 3072) tensor
        
        # for the query, we'll add an <image> token at the beginning, which we'll tokenize to -200
        query_tok = self.tokenizer(query, return_tensors="pt", padding="max_length")
        query_tok = torch.cat([torch.tensor([-200]).unsqueeze(0), query_tok["input_ids"]], dim=1)

        concat_embd, attn_mask = self.process_tensors(query_tok, lm_img_features, self.embedding_layer)

        output_ids = self.decoder.generate(inputs_embeds=concat_embd, attention_mask=attn_mask, max_length=2000)
        
        if targets is not None:
            loss = self.decoder(**query_tok, labels=targets).loss
            return output_ids, loss
        else:
            return output_ids

    def process_tensors(input_ids, image_features, embedding_layer):
        # Find the index of -200 in input_ids
        split_index = (input_ids == -200).nonzero(as_tuple=True)[1][0]

        # Split the input_ids at the index found, excluding -200
        input_ids_1 = input_ids[:, :split_index]
        input_ids_2 = input_ids[:, split_index + 1 :]

        # Convert input_ids to embeddings
        embeddings_1 = embedding_layer(input_ids_1)
        embeddings_2 = embedding_layer(input_ids_2)

        device = image_features.device
        token_embeddings_part1 = embeddings_1.to(device)
        token_embeddings_part2 = embeddings_2.to(device)

        # Concatenate the token embeddings and image features
        concatenated_embeddings = torch.cat(
            [token_embeddings_part1, image_features, token_embeddings_part2], dim=1
        )

        # Create the corrected attention mask
        attention_mask = torch.ones(
            concatenated_embeddings.shape[:2], dtype=torch.long, device=device
        )
        return concatenated_embeddings, attention_mask