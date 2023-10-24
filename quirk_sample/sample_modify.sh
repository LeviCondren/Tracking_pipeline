cp /eos/user/q/qsha/work/track/Event_files/*/mass_500_Lambda_500/*.csv /eos/user/q/qsha/work/track/QuirkTracking-ML/Examples/QuirkTracking/datasets/50layers/Lambda500_dataset/mix
cp /eos/user/q/qsha/work/track/Event_files/*/mass_500_Lambda_500/Bgd/*.csv /eos/user/q/qsha/work/track/QuirkTracking-ML/Examples/QuirkTracking/datasets/50layers/Lambda500_dataset/bkg
cp /eos/user/q/qsha/work/track/Event_files/*/mass_500_Lambda_500/Quirk/*.csv /eos/user/q/qsha/work/track/QuirkTracking-ML/Examples/QuirkTracking/datasets/50layers/Lambda500_dataset/quirk

cd /eos/user/q/qsha/work/track/QuirkTracking-ML/Examples/QuirkTracking/datasets/50layers/Lambda500_dataset
source name.sh
source rename_230811.sh