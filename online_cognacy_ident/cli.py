import argparse
import csv
import random
import time

from json.decoder import JSONDecodeError

import pyclts

from online_cognacy_ident.clustering import cluster
from online_cognacy_ident.dataset import (
    Dataset, CLDFDataset, PairsDataset, DatasetError,
    write_clusters, write_cldf_clusters)
from online_cognacy_ident.evaluation import calc_f_score
from online_cognacy_ident.model import save_model, load_model, ModelError
from online_cognacy_ident.phmm import train_phmm, apply_phmm
from online_cognacy_ident.pmi import train_pmi, apply_pmi

sound_classes = pyclts.soundclasses.SOUNDCLASS_SYSTEMS



def number_in_interval(number, type, interval):
    """
    Typecast the number to an int or a float and ensure that it is within the
    specified interval. Raise an ArgumentTypeError otherwise.

    Helper for Cli's ArgumentParser instance.
    """
    try:
        number = type(number)
    except ValueError:
        raise argparse.ArgumentTypeError(
                'invalid {} value: {!r}'.format(type.__name__, number))

    if number < interval[0] or number > interval[1]:
        raise argparse.ArgumentTypeError(
                '{} not in interval [{}; {}]'.format(number, interval[0], interval[1]))

    return number



class TrainCli:
    """
    Handles the user input, invokes the necessary functions, and takes care of
    exiting the program for training pmi/phmm models.

    Usage:
        if __name__ == '__main__':
            cli = TrainCli()
            cli.run()
    """

    def __init__(self):
        """
        Init the argparse parser.
        """
        self.parser = argparse.ArgumentParser(add_help=False, description=(
            'train a pmi or phmm model on a dataset'))

        self.parser.add_argument(
            'algorithm',
            choices=['pmi', 'phmm'],
            help='which of the two algorithms to use')
        self.parser.add_argument(
            'dataset',
            help=(
                'path to a dataset to train on; '
                'this could be either a csv/tsv file (as in the datasets dir) '
                'or a word pairs file (as in the training_data dir)'))
        self.parser.add_argument(
            'output',
            help='path where to store the trained model')

        algo_args = self.parser.add_argument_group('optional arguments - algorithm')
        algo_args.add_argument(
            '-c', '--initial-cutoff',
            type=lambda x: number_in_interval(x, float, [0, 1]),
            default=0.5,
            help=(
                'initial Levenshtein distance cutoff; '
                'should be within the interval [0.0; 1.0]; '
                'word pairs with normalised edit distance '
                'above this threshold are ignored'))
        algo_args.add_argument(
            '-a', '--alpha',
            type=lambda x: number_in_interval(x, float, [0.5, 1]),
            default=0.75,
            help=(
                'α, EM hyperparameter; should be within the interval [0.5; 1]; '
                'the default value is 0.75'))
        algo_args.add_argument(
            '-m', '--batch-size',
            type=lambda x: number_in_interval(x, int, [1, float('inf')]),
            default=256,
            help=(
                'm, EM hyperparameter; should be a positive integer; '
                'the default value is 256'))
        algo_args.add_argument(
            '-r', '--random-seed',
            type=int,
            default=42,
            help=(
                'integer to use as seed for python\'s random module; '
                'the default value is 42'))

        io_args = self.parser.add_argument_group('optional arguments - input/output')
        io_args.add_argument(
            '--dataset-type',
            choices=['standard', 'pairs', 'cldf'],
            default='pairs',
            help=(
                'pairs (the default) refers to the specific format used '
                'for the datasets in the training_data dir; '
                'standard refers to the csv/tsv format used in '
                'the datasets dir'))
        io_args.add_argument(
            '--csv-dialect',
            choices=csv.list_dialects(),
            help=(
                'the csv dialect to use for reading the dataset; '
                'the default is to look at the file extension '
                'and use excel for .csv and excel-tab for .tsv'))
        io_args.add_argument(
            '-i', '--ipa',
            action='store_true',
            help=(
                'convert input transcriptions from IPA to another sound class model; '
                'by default these are assumed to be in target transcription'))
        io_args.add_argument(
            '-s', '--sound-class-model',
            choices=sound_classes,
            default='asjp',
            help=(
                'convert IPA input to this sound class model; '
                'ignored when input is not IPA. '
                '(default: asjp)'))

        other_args = self.parser.add_argument_group('optional arguments - other')
        other_args.add_argument(
            '-h', '--help',
            action='help',
            help='show this help message and exit')
        other_args.add_argument(
            '-t', '--time',
            action='store_true',
            help='show total training time at the end')


    def run(self, raw_args=None):
        """
        Parse the given args (if these are None, default to parsing sys.argv,
        which is what you would want unless you are unit testing).
        """
        args = self.parser.parse_args(raw_args)

        random.seed(args.random_seed)
        start_time = time.time()

        if args.ipa:
            sc = pyclts.SoundClasses(args.sound_class_model)
            def transform(sound):
                try:
                    return sc[sound]
                except KeyError:
                    return '0'
        else:
            transform = None

        try:
            if args.dataset_type == 'pairs':
                dataset = PairsDataset(args.dataset, transform=transform)
            elif args.dataset_type == 'cldf':
                dataset = CLDFDataset(args.dataset, transform=transform)
            else:
                dataset = Dataset(args.dataset, args.csv_dialect, transform=transform)
        except DatasetError as err:
            self.parser.error(str(err))

        print(
            'training {} on {}, conversion={}{}{}, m={!s}, α={:.2f}'.format(
                args.algorithm.upper(), args.dataset,
                'ipa→' if args.ipa else '(', args.sound_class_model, '' if args.ipa else ')',
                args.batch_size, args.alpha))

        train_func = train_phmm if args.algorithm == 'phmm' else train_pmi
        model = train_func(
                    dataset, initial_cutoff=args.initial_cutoff,
                    alpha=args.alpha, batch_size=args.batch_size)
        try:
            save_model(args.output, args.algorithm, model)
        except ModelError as err:
            self.parser.error(str(err))

        if args.time:
            print('training time: {:.2f} sec'.format(time.time() - start_time))



class RunCli:
    """
    Handles the user input, invokes the necessary classes and functions, and
    takes care of exiting the programme for running the cognacy identification
    algorithms.

    Usage:
        if __name__ == '__main__':
            cli = RunCli()
            cli.run()
    """

    def __init__(self):
        """
        Init the argparse parser.
        """
        self.parser = argparse.ArgumentParser(add_help=False, description=(
            'run the pmi or phmm online cognacy identification algorithm '
            'on a dataset'))

        self.parser.add_argument(
            'model',
            help='path to a trained model file')
        self.parser.add_argument(
            'dataset',
            help='path to a dataset to run cognacy identification on')
        self.parser.add_argument(
            '--cluster-method',
            default="infomap",
            help='clustering method to use (default: infomap)')

        io_args = self.parser.add_argument_group('optional arguments - input/output')
        io_args.add_argument(
            '--dialect-input',
            choices=csv.list_dialects(),
            help=(
                'the csv dialect to use for reading the dataset; '
                'the default is to look at the file extension '
                'and use excel for .csv and excel-tab for .tsv'))
        io_args.add_argument(
            '--dialect-output',
            choices=csv.list_dialects(), default='excel-tab',
            help=(
                'the csv dialect to use for writing the output; '
                'the default is excel-tab'))
        io_args.add_argument(
            '-o', '--output',
            help=(
                'path where to write the identified cognate classes; '
                'defaults to stdout'))
        io_args.add_argument(
            '-i', '--ipa',
            action='store_true',
            help=(
                'convert input transcriptions from IPA to another sound class model; '
                'by default these are assumed to be in target transcription'))
        io_args.add_argument(
            '-s', '--sound-class-model',
            choices=sound_classes,
            default='asjp',
            help=(
                'convert IPA input to this sound class model; '
                'ignored when input is not IPA. '
                '(default: asjp)'))
        io_args.add_argument(
            '-e', '--evaluate',
            action='store_true',
            help=(
                'evaluate the output against the input dataset and '
                'print the resulting F-score; this will fail '
                'if the input dataset does not include cognate classes'))

        other_args = self.parser.add_argument_group('optional arguments - other')
        other_args.add_argument(
            '-h', '--help',
            action='help',
            help='show this help message and exit')
        other_args.add_argument(
            '-t', '--time',
            action='store_true',
            help='show total running time at the end')


    def run(self, raw_args=None):
        """
        Parse the given args (if these are None, default to parsing sys.argv,
        which is what you would want unless you are unit testing).
        """
        args = self.parser.parse_args(raw_args)
        output_cldf_dataset = False

        start_time = time.time()

        if args.ipa:
            sc = pyclts.SoundClasses(args.sound_class_model)
            def transform(sound):
                return sc[sound]
        else:
            transform = None

        try:
            try:
                dataset = CLDFDataset(args.dataset, transform=transform)
                output_cldf_dataset = True
            except JSONDecodeError:
                dataset = Dataset(args.dataset, args.dialect_input, transform=transform)
            algorithm, model = load_model(args.model)
        except (DatasetError, ModelError) as err:
            self.parser.error(str(err))

        print(
            'running {} on {}, conversion={}{}{}'.format(
                args.model, args.dataset,
                'ipa→' if args.ipa else '(', args.sound_class_model, '' if args.ipa else ')'))

        if algorithm == 'phmm':
            scores = apply_phmm(dataset, *model)
        else:
            scores = apply_pmi(dataset, model)

        clusters = cluster(dataset, scores, method=args.cluster_method)
        if output_cldf_dataset:
            write_cldf_clusters(clusters, args.output, dataset.dataset)
        else:
            write_clusters(clusters, args.output, args.dialect_output)

        if args.time:
            print('running time: {:.2f} sec'.format(time.time() - start_time))

        if args.evaluate:
            score = calc_f_score(dataset.get_clusters(), clusters)
            print('f-score: {:.4f}'.format(score))



class EvalCli:
    """
    Handles the user input, invokes the corresponding code, and takes care of
    exiting the programme for evaluating the output of cognacy identification
    algorithms.

    Usage:
        if __name__ == '__main__':
            cli = EvalCli()
            cli.run()
    """

    def __init__(self):
        """
        Init the argparse parser.
        """
        self.parser = argparse.ArgumentParser(add_help=False, description=(
            'evaluate the cognates clustering of a dataset '
            'against the same data\'s gold-standard cognate classes'))

        self.parser.add_argument('dataset_true', help=(
            'path to the dataset containing the gold-standard cognate classes'))
        self.parser.add_argument('dataset_pred', help=(
            'path to the dataset containing the predicted cognate classes'))

        csv_args = self.parser.add_argument_group('optional arguments')
        csv_args.add_argument(
            '--type-true',
            choices=["csv", "cldf"],
            default="csv",
            help=('Is the reference dataset a CSV or a CLDF dataset?'))
        csv_args.add_argument(
            '--type-pred',
            choices=["csv", "cldf"],
            default="csv",
            help=('Is the prediction dataset a CSV or a CLDF dataset?'))

        csv_args.add_argument('--dialect-true',
            choices=csv.list_dialects(), help=(
                'the csv dialect to use for reading the dataset '
                'that contains the gold-standard cognate classes; '
                'the default is to look at the file extension '
                'and use excel for .csv and excel-tab for .tsv'))
        csv_args.add_argument('--dialect-pred',
            choices=csv.list_dialects(), help=(
                'the csv dialect to use for reading the dataset '
                'that contains the predicted cognate classes; '
                'the default is to look at the file extension '
                'and use excel for .csv and excel-tab for .tsv'))

        other_args = self.parser.add_argument_group('optional arguments - other')
        other_args.add_argument('-h', '--help', action='help', help=(
            'show this help message and exit'))


    def run(self, raw_args=None):
        """
        Parse the given args (if these are None, default to parsing sys.argv,
        which is what you would want unless you are unit testing), invoke the
        evaluation function and print its output.
        """
        args = self.parser.parse_args(raw_args)

        try:
            if args.type_true == 'csv':
                dataset_true = Dataset(args.dataset_true, args.dialect_true)
            elif args.type_true == 'cldf':
                dataset_true = CLDFDataset(args.dataset_true)
            if args.type_true == 'csv':
                dataset_pred = Dataset(args.dataset_pred, args.dialect_pred)
            elif args.type_true == 'cldf':
                dataset_pred = CLDFDataset(args.dataset_pred)
        except DatasetError as err:
            self.parser.error(str(err))

        score = calc_f_score(dataset_true.get_clusters(), dataset_pred.get_clusters())
        print('{:.4f}'.format(score))
