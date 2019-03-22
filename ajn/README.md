5.2. Minibatch - Average gradient of the loss by time-step:

To execute the code you can use the argument --grad_minibatch. This option is disabled when this option is 0 and enable when it's 1
s_exec python3 ptb-lm.py --model=RNN --optimizer=SGD_LR_SCHEDULE --initial_lr=1 --batch_size=20 --seq_len=35 --hidden_size=200 --num_layers=2 
--dp_keep_prob=0.35 --save_best  --num_epochs=35 --grad_minibatch=1
