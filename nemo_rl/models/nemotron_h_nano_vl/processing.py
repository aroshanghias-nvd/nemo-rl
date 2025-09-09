from transformers import ProcessorMixin, BatchEncoding


class NemotronNanoVLV2Processor(ProcessorMixin):

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "NemotronNanoVLV2ImageProcessor"
    tokenizer_class = "PreTrainedTokenizerFast"

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        super().__init__(image_processor, tokenizer)

        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")

    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        tokenizer_kwargs, image_processor_kwargs = {}, {}
        if kwargs:
            tokenizer_kwargs = {k: v for k, v in kwargs.items() if k not in self.image_processor._valid_processor_keys}
            image_processor_kwargs = {
                k: v for k, v in kwargs.items() if k in self.image_processor._valid_processor_keys
            }

        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            encoding = self.tokenizer(text, return_tensors=return_tensors, **tokenizer_kwargs)

        if images is not None:
            image_features = self.image_processor(images, return_tensors=return_tensors, **image_processor_kwargs)

        # TODO(jseppanen): insert image context tokens, based on image_num_patches

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features.pixel_values
            return encoding
        elif text is not None:
            return encoding
        else:
            return BatchEncoding(data=dict(**image_features), tensor_type=return_tensors)

    def conversation_preprocessor(self, messages):
        return messages

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, return_tensors=None, return_dict=None, **kwargs):
        return self.tokenizer.apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt, return_tensors=return_tensors, return_dict=return_dict, **kwargs)
