CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" OMP_NUM_THREADS=4 \
nohup python -m torch.distributed.launch --nproc_per_node 8 --master_port 12348 --master_addr localhost main.py 1> log/0628_1646.log 2>&1 &


"""
tmux 인스톨 하고 그냥 터미널처럼
사용가능해요
터미널 꺼도
일반 bach 창에서
tmux ls 이렇게 세션 체크 할수 있고
그럼 그 윈도우가 현재 붙어있는지 (볼수 있는상태) 떨어져있는지 (못보는 상태) 체크해서
tmux attach -t [세션명] 명령어로 그 세션 다시 켜서 확인 가능
"""