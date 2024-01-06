#!/bin/bash

python3 generate.py     /home/gb/hzy/second-article/mask-our-model-3090-tiaoshi/data-bin-the-same-as-EMMT \
				--path results/mmtimg1/model.pt \
				--source-lang en --target-lang zh \
				--beam 5 \
				--num-workers 12 \
				--batch-size 128 \
				--results-path results \
				--remove-bpe \
#				--fp16 \