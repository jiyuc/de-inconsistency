for T in mclass
do
    TEST=~/PycharmProjects/de-inconsistency/model/data/test.txt
    echo $TEST
    python g_evaluation.py $TEST $T
done
          
