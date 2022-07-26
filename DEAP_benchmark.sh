for token in QKV
do
    for i in {1..32}
    do
      echo "Benchmark on subject: $i, start ..."
      CUDA_VISIBLE_DEVICES=1 python DEAP_test_gen.py --subject $i --nn_token $token --expType 2 --valMode random
  #   echo "Training on $token"
      echo "Benchmark on subject: $i, finished"
    done
done
