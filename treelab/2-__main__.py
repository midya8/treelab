import argparse,json,sys
from pathlib import Path
from . import experiment,tree


def strict_load(path):
    def bad(s):raise ValueError('nonfinite JSON constant '+s)
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('duplicate JSON key '+k)
            d[k]=v
        return d
    return json.loads(Path(path).read_text(),parse_constant=bad,object_pairs_hook=pairs)


def main():
    p=argparse.ArgumentParser(description='CART/pruning with explicit data and protocol limits');s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('audit');a.add_argument('--output',required=True)
    s.add_parser('demo');a=s.add_parser('predict');a.add_argument('--model',required=True);a.add_argument('--input',required=True)
    a=s.add_parser('fit');a.add_argument('--input',required=True);a.add_argument('--output',required=True);a.add_argument('--min-leaf',type=int,default=5);a.add_argument('--max-depth',type=int,default=20)
    a=s.add_parser('prune');a.add_argument('--model',required=True);a.add_argument('--alpha',type=float,required=True);a.add_argument('--output',required=True)
    args=p.parse_args()
    try:
        if args.command=='audit':obj=experiment.run(args.output)
        elif args.command=='demo':obj=strict_load(Path(__file__).resolve().parent/'artifacts/report.json')
        elif args.command in ('fit','prune'):
            if args.command=='fit':
                raw=strict_load(args.input)
                if not isinstance(raw,dict) or set(raw)!={'features','targets'}:raise ValueError('fit input must contain only features and targets')
                model=tree.fit(raw['features'],raw['targets'],args.min_leaf,args.max_depth)
            else:model=tree.prune(strict_load(args.model),args.alpha)
            Path(args.output).write_text(json.dumps(model,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
            risk,leaves=tree.subtree(model['root']);obj={'leaves':leaves,'fit_mse':risk/model['fit_rows']}
        else:
            model=strict_load(args.model);raw=strict_load(args.input)
            if not isinstance(raw,dict) or set(raw)!={'features'}:raise ValueError('input must contain only features')
            pred,paths=tree.predict(model,raw['features'],paths=True);obj={'predictions':pred.tolist(),'paths':paths}
        print(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False))
    except (ValueError,TypeError,KeyError,OSError,ArithmeticError) as e:p.exit(2,str(e)+'\n')

if __name__=='__main__':main()
