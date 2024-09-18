cd ~/SMABalancingGroups
. /home/aptikal/gauffrea/SMABalancingGroups/envSMAB2/bin/activate
git pull
nohup python3 train_cl.py --config configC3-0.yaml &
wait
