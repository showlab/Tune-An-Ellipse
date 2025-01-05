

REF=refcocog
SPLITS="test val"
SPLITBY="umd"
REFCOCO_ROOT="/PATH/TO/RefCOCO"


for SPLIT in ${SPLITS}
do

    echo REF=${REF} SPLIT=${SPLIT} SPLITBY=${SPLITBY} TOPK=${TOPK}
    
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run_ref.py --root ${REFCOCO_ROOT} \
        --workspace workspace/${REF}_${SPLIT} \
        --reftype ${REF} --split ${SPLIT} --splitby ${SPLITBY} \
        --process_num 16

    python eval_ref.py \
        --workspace workspace/${REF}_${SPLIT} \
        --root ${REFCOCO_ROOT} --reftype ${REF} --split ${SPLIT} --splitby ${SPLITBY}

done










