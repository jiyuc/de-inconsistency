for T in mclass
do
    TEST=/model/data/test.txt
    echo $TEST
    python g_evaluation.py $TEST $T
done
          
