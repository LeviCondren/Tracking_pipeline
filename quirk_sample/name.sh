#!/bin/bash

# 遍历当前文件夹中的所有文件
for file in *-particle.csv; do
    # 检查文件是否存在
    if [ -e "$file" ]; then
        # 获取文件名和扩展名
        filename=$(basename "$file")
        extension="${filename##*.}"
        
        # 获取文件名（不包含扩展名）和新文件名
        basename="${filename%-particle.csv}"
        new_filename="${basename}-particles.csv"
        
        # 重命名文件
        mv "$file" "$new_filename"
        echo "Renamed $file to $new_filename"
    fi
done

