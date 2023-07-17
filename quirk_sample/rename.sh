#!/bin/bash

# 复制并重命名 Event_0_QuirkQuirkBgd_Mass_500.csv 文件
for ((i=0; i<=100; i++))
do
    # 构造新的文件名
    new_filename=$(printf "event000001%03d-particles.csv" "$((i))")
    
    # 复制并重命名文件
    cp "./Event_files/$i/Event_${i}_QuirkQuirkBgd_Mass_500.csv" "./Event_files/$new_filename"
done

# 复制并重命名 Hits_Event0_QuirkQuirk_Lambda_100_.csv 文件
for ((i=0; i<=100; i++))
do
    # 构造新的文件名
    new_filename=$(printf "event000001%03d-hits.csv" "$((i))")
    
    # 复制并重命名文件
    cp "./Event_files/$i/Hits_Event${i}_QuirkQuirk_Lambda_100_.csv" "./Event_files/$new_filename"
done

# 修改带有 "hits" 名的 CSV 文件的第一行
find Event_files -type f -name "*-hits.csv" -exec sed -i '1s/"HitID","r\[cm\]","phi","z\[cm\]","layer_id","particle_id"/hit_id,r,phi,z,layer_id,particle_id/' {} \;
