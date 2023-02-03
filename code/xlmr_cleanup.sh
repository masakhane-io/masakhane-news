for LANG in amh eng fra hau ibo lin orm pcm run sna swa yor
do
  rm -rf ${LANG}_xlmrbase/pytorch_model.bin
  rm -rf ${LANG}_xlmrbase/tokenizer.json
  rm -rf ${LANG}_xlmrbase/tokenizer_config.json
  rm -rf ${LANG}_xlmrbase/training_args.bin
  rm -rf ${LANG}_xlmrbase/config.json
  rm -rf ${LANG}_xlmrbase/special_tokens_map.json
  done
