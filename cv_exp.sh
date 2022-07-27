for i in {1..15}
do
  echo "CV benchmark on subject: $i, start ..."
  CUDA_VISIBLE_DEVICES=1 python exp_5CV_SEED.py --subject $i
  echo "CV benchmark on subject: $i, finished"
done
