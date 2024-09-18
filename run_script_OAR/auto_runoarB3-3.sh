cd ~/SMABalancingGroups
. /home/aptikal/gauffrea/SMABalancingGroups/envSMAB2/bin/activate
git pull
nohup python3 train_cl.py --config configB3-3.yaml &
wait
