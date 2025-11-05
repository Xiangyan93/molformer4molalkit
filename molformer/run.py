import os
import optuna
from optuna.samplers import TPESampler
from molformer.args import TrainArgs, OptunaArgs
from molformer.checkpoints import AVAILABLE_CHECKPOINTS
from molformer.evaluator import Evaluator
from molformer.molformer import MolFormer


def molformer_cv(arguments=None):
    args = TrainArgs().parse_args(arguments)
    model = MolFormer(
        save_dir=args.save_dir,
        pretrained_path=args.pretrained_path,
        task_type=args.task_type,
        num_tasks=len(args.targets_columns),
        n_head=args.n_head, 
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        d_dropout=args.d_dropout,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        num_feats=args.num_feats,
        ensemble_size=args.ensemble_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.n_jobs,
        seed=args.seed
    )
    evaluator = Evaluator(
        save_dir=args.save_dir,
        dataset=args.dataset,
        model=model,
        task_type=args.task_type,
        metrics=args.metrics,
        cross_validation=args.cross_validation,
        n_splits=args.n_splits,
        split_type=args.split_type,
        split_sizes=args.split_sizes,
        num_folds=args.num_folds,
        seed=args.seed,
        verbose=True,
    )
    if args.separate_test_path is not None:
        evaluator.run_external(args.dataset_test)
    else:
        evaluator.run_cross_validation()


def molformer_optuna(arguments=None):
    args = OptunaArgs().parse_args(arguments)
    os.makedirs(args.save_dir, exist_ok=True)
    def objective(trial):
        params = {
            'pretrained_path': trial.suggest_categorical('pretrained_path', AVAILABLE_CHECKPOINTS),
            # 'n_head': trial.suggest_int('n_head', 4, 20, step=4),
            'n_layer': trial.suggest_int('n_layer', 2, 20, step=1),
            # 'n_embd': trial.suggest_categorical('n_embd', [128, 256, 512, 768]),
            'd_dropout': trial.suggest_float('d_dropout', 0.0, 0.4, step=0.05),
            'dropout': trial.suggest_float('dropout', 0.0, 0.4, step=0.05),
            'learning_rate': trial.suggest_float('learning_rate', 3e-6, 3e-4, log=True),
            # 'num_feats': trial.suggest_categorical('num_feats', [16, 32, 64, 128]),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'epochs': trial.suggest_categorical('epochs', [10, 30, 50, 100, 200]),
        }
        model = MolFormer(save_dir='%s/trial-%d' % (args.save_dir, trial.number), 
                          task_type=args.task_type, num_tasks=len(args.targets_columns),
                          num_workers=args.n_jobs, seed=args.seed,
                          **params)
        evaluator = Evaluator(save_dir='%s/trial-%d' % (args.save_dir, trial.number),
                            dataset=args.dataset,
                            model=model,
                            task_type=args.task_type,
                            metrics=args.metrics,
                            cross_validation=args.cross_validation,
                            n_splits=args.n_splits,
                            split_type=args.split_type,
                            split_sizes=args.split_sizes,
                            num_folds=args.num_folds,
                            seed=args.seed)
        if args.separate_val_path is not None:
            if args.separate_test_path is not None:
                evaluator1 = Evaluator(save_dir='%s/trial-%d' % (args.save_dir, trial.number),
                                        dataset=args.dataset_train_val,
                                        model=model,
                                        task_type=args.task_type,
                                        metrics=args.metrics,
                                        cross_validation=args.cross_validation,
                                        n_splits=args.n_splits,
                                        split_type=args.split_type,
                                        split_sizes=args.split_sizes,
                                        num_folds=args.num_folds,
                                        seed=args.seed)
                evaluator1.run_external(args.dataset_test, name='test')
                evaluator.run_external(args.dataset_test, name='test_train_only')
            return evaluator.run_external(args.dataset_val, name='val')
        else:
            if args.separate_test_path is not None:
                evaluator.run_external(args.dataset_test, name='test')
            return evaluator.run_cross_validation()

    study = optuna.create_study(
        study_name="optuna-study",
        sampler=TPESampler(seed=args.seed),
        storage="sqlite:///%s/optuna.db" % args.save_dir,
        load_if_exists=True,
        direction='minimize' if args.task_type == 'regression' else 'maximize'
    )
    n_to_run = args.n_trials - len(study.trials)
    if n_to_run > 0:
        study.optimize(objective, n_trials=n_to_run)
