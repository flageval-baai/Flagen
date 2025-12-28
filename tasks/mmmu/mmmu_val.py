
dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt="Let's think step by step and put the letter of your final choice after 'Answer: '",
    ),
    data_root="/share/project/mmdataset/MMMU/validation",
    name="mmmu_val",
)

evaluator = dict(type="MmmuEvaluator")
