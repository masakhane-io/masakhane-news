for sample in 5 10 20 50
do

  for LANG in amh eng fra hau ibo lin orm pcm run sna swa yor
  do
    rm -rf ${sample}_${LANG}_labse/pytorch_model.bin
    rm -rf ${sample}_${LANG}_labse/tokenizer.json
    rm -rf ${sample}_${LANG}_labse/tokenizer_config.json
    rm -rf ${sample}_${LANG}_labse/training_args.bin
    rm -rf ${sample}_${LANG}_labse/config.json
    rm -rf ${sample}_${LANG}_labse/special_tokens_map.json
    rm -rf ${sample}_${LANG}_labse/vocab.txt
    done
  done