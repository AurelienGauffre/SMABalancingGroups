cd ~/SMABalancingGroups
. /home/aptikal/gauffrea/SMABalancingGroups/envSMAB2/bin/activate
git pull
nohup python3 pretrain.py --config configZ1.yaml &
wait
