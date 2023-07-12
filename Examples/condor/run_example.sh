#!/bin/bash
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup rucio
dir=/afs/cern.ch/user/q/qsha/mg5batch
model=yy2l_bkg

output=run_100
counter=`echo $output |awk -F\_ '{print $NF}'`

echo $counter
#random number seed
let rnd=1651+$counter*912
now=$(date +"%T")
cd /tmp
pwd
cp -r $dir/$model ./
mkdir -vp $model/Events
echo $rnd
sed -i 's/0   = iseed/'$rnd' = iseed/g' $model/Cards/run_card.dat
./$model/bin/generate_events $output -f --nb_core=1 --laststep=pythia

gunzip -d $model/Events/$output/*.gz
cp -r $model/Events/$output /eos/user/q/qsha/yy2l/$output
hepmc=`ls $model/Events/$output/*.hepmc`

echo "hepmc" $hepmc

source $dir/delphes.sh $hepmc $output

cp *root /eos/user/q/qsha/yy2l





