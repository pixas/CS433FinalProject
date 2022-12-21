batch_size_list=(16 32 64 128)
benchmark_store_file_list=(
    "/home/group14/CS433FinalProject/task1/target/benchmark/benchmark_b16.txt" \
    "/home/group14/CS433FinalProject/task1/target/benchmark/benchmark_b32.txt" \
    "/home/group14/CS433FinalProject/task1/target/benchmark/benchmark_b64.txt" \
    "/home/group14/CS433FinalProject/task1/target/benchmark/benchmark_b128.txt" \
)

/home/group14/CS433FinalProject/task1/target/bin/oracle > /home/group14/CS433FinalProject/task1/target/output/oracle_predictions.txt
echo "oracle predictions generated"

for((i=1;i<=4;i++));  
do
echo "batch size: ${batch_size_list[i-1]}"
/home/group14/CS433FinalProject/task1/target/bin/inference ${batch_size_list[i-1]} > ${benchmark_store_file_list[i-1]}
python3 src/validate.py
echo "----------"
done