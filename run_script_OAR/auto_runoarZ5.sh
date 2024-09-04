cd ~/SMABalancingGroups
. /home/aptikal/gauffrea/SMABalancingGroups/envSMAB2/bin/activate
git pull
nohup python3 pretrain.py --config configZ5.yaml &
wait
