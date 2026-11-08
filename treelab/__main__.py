import argparse,gzip,json,sys
from pathlib import Path
from . import bagging,experiment,sweep,tree


def strict_load(path):
    def bad(s):raise ValueError('nonfinite JSON constant '+s)
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('duplicate JSON key '+k)
            d[k]=v
        return d
    raw=Path(path).read_bytes()
    if str(path).endswith('.gz'):raw=gzip.decompress(raw)
    return json.loads(raw,parse_constant=bad,object_pairs_hook=pairs)


def main():
    p=argparse.ArgumentParser(description='CART/pruning with explicit data and protocol limits');s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('audit');a.add_argument('--output',required=True)
    s.add_parser('demo');a=s.add_parser('predict');a.add_argument('--model',required=True);a.add_argument('--input',required=True)
    a=s.add_parser('fit');a.add_argument('--input',required=True);a.add_argument('--output',required=True);a.add_argument('--min-leaf',type=int,default=5);a.add_argument('--max-depth',type=int,default=20)
    a=s.add_parser('prune');a.add_argument('--model',required=True);a.add_argument('--alpha',type=float,required=True);a.add_argument('--output',required=True)
    s.add_parser('bag-demo')
    a=s.add_parser('bag-audit');a.add_argument('--output',required=True)
    a=s.add_parser('bag-fit');a.add_argument('--input',required=True);a.add_argument('--output',required=True);a.add_argument('--trees',type=int,default=32);a.add_argument('--seed',type=int,default=20261108);a.add_argument('--min-leaf',type=int,default=5);a.add_argument('--max-depth',type=int,default=20)
    a=s.add_parser('bag-predict');a.add_argument('--model',required=True);a.add_argument('--input',required=True);a.add_argument('--trees',type=int)
    a=s.add_parser('bag-oob');a.add_argument('--model',required=True);a.add_argument('--input',required=True)
    s.add_parser('sweep-demo')
    a=s.add_parser('sweep-audit');a.add_argument('--output',required=True)
    a=s.add_parser('sweep');a.add_argument('--model',required=True);a.add_argument('--input',required=True);a.add_argument('--trees',type=int)
    args=p.parse_args()
    try:
        if args.command=='sweep-demo':obj=strict_load(Path(__file__).resolve().parent/'artifacts/sweep-report.json')
        elif args.command=='sweep-audit':obj=sweep.run(args.output)
        elif args.command=='sweep':
            raw=strict_load(args.input)
            if not isinstance(raw,dict) or set(raw)!={'row','feature','low','high'}:raise ValueError('sweep input must contain only row, feature, low, high')
            obj=sweep.response(strict_load(args.model),raw['row'],raw['feature'],raw['low'],raw['high'],args.trees)
        elif args.command=='bag-demo':obj=strict_load(Path(__file__).resolve().parent/'artifacts/bagging-report.json')
        elif args.command=='bag-audit':obj=bagging.run(args.output)
        elif args.command in ('bag-fit','bag-predict','bag-oob'):
            raw=strict_load(args.input)
            fields={'features'} if args.command=='bag-predict' else {'features','targets'}
            if not isinstance(raw,dict) or set(raw)!=fields:raise ValueError('bag input fields must be '+str(sorted(fields)))
            if args.command=='bag-fit':
                model=bagging.fit(raw['features'],raw['targets'],args.trees,args.seed,args.min_leaf,args.max_depth)
                Path(args.output).write_text(json.dumps(model,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n');obj={'trees':len(model['trees']),'groups':len(model['groups'])}
            else:
                model=strict_load(args.model)
                obj={'predictions':bagging.predict(model,raw['features'],args.trees).tolist()} if args.command=='bag-predict' else bagging.oob(model,raw['features'],raw['targets'],[len(model['trees'])])
        elif args.command=='audit':obj=experiment.run(args.output)
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
