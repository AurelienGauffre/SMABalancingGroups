cd ~/SMABalancingGroups
. /home/aptikal/gauffrea/SMABalancingGroups/envSMAB2/bin/activate
git pull
nohup python3 train_cl.py --config configA6.yaml &
wait