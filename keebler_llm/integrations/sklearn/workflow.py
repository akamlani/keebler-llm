from sklearn.pipeline import Pipeline, make_pipeline 

def generate_pipeline(params:dict, format_pandas:bool=True) -> Pipeline:
    make_name = lambda prefix, cls: "_".join([prefix, cls.__name__.lower()])
    pipeline  = Pipeline(steps = [
        (make_name(k, v['cls']), v['cls'](**v['params']))
        for k,v in params.items()
    ])
    if format_pandas:
        pipeline = pipeline.set_output(transform='pandas')

    return pipeline



