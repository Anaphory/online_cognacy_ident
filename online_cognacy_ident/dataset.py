from collections import defaultdict, namedtuple

import csv
import itertools
import os.path
import sys

import pycldf
import pyclts
from segments import Tokenizer, Profile

from online_cognacy_ident.align import normalized_levenshtein

bipa = pyclts.TranscriptionSystem("bipa")
tokenizer = Tokenizer()


"""
Dict mapping the column names that the Dataset class looks for to lists of
possible variants. Used in Dataset._read_header().
"""
RECOGNISED_COLUMN_NAMES = {
    'doculect': ['doculect', 'language', 'lang'],
    'concept': ['concept', 'gloss'],
    'asjp': ['asjp', 'transcription'],
    'cog_class': ['cog_class', 'cognate_class', 'cognate class', 'cogid']
}



"""
The named tuple used in the return values of the get_words, get_concepts, and
get_clusters methods of Dataset objects.
"""
Word = namedtuple('Word', ['doculect', 'concept', 'sc', 'id'])



class DatasetError(ValueError):
    """
    Raised when something goes wrong with reading a dataset.
    """
    pass



class Dataset:
    """
    Handles dataset reading. It is assumed that the dataset would be a csv/tsv
    file that contains at least one of the columns for each list of recognised
    column names.

    Usage:

        try:
            dataset = Dataset(path)
            for concept, words in dataset.concepts():
                print(concept)
        except DatasetError as err:
            print(err)
    """

    def __init__(self, path, dialect=None, transform=None):
        """
        Set the instance's props. Raise a DatasetError if the given file path
        does not exist. 

        The dialect arg should be either a string identifying one of the csv
        dialects or None, in which case the dialect is inferred based on the
        file extension. Raise a ValueError if the given dialect is specified
        but unrecognised.

        If is_ipa is set, assume that the transcriptions are in IPA and convert
        them into some other sound class model.
        """
        if not os.path.exists(path):
            raise DatasetError('Could not find file: {}'.format(path))

        if dialect is None:
            dialect = 'excel-tab' if path.endswith('.tsv') else 'excel'
        elif dialect not in csv.list_dialects():
            raise ValueError('Unrecognised csv dialect: {!s}'.format(dialect))

        self.path = path
        self.dialect = dialect
        self.transform = transform

        self.alphabet = None


    def _read_header(self, line, exclude=['cog_class']):
        """
        Return a {column name: index} dict, excluding the columns listed in the
        second func arg.

        Raise a DatasetError if not all required columns can be recovered.
        """
        d = {}

        column_names = {column: names
                for column, names in RECOGNISED_COLUMN_NAMES.items()
                if column not in exclude}

        for index, heading in enumerate(line):
            heading = heading.lower()
            for column, recognised_names in column_names.items():
                if heading in recognised_names:
                    d[column] = index
                    break

        for column in column_names.keys():
            if column not in d:
                raise DatasetError('Could not find the column for {}'.format(column))

        return d


    def _read_asjp(self, raw_trans):
        """
        Process a raw transcription value into a sound class transcription, eg. ASJP:
        (1) if the input string consists of multiple comma-separated entries,
        remove all but the first one;
        (2) remove whitespace chars (the symbols +, - and _ are also considered
        whitespace and removed);
        (3) if this is an IPA dataset, convert the string to ASJP;
        (4) remove some common non-ASJP offender symbols.

        Helper for the _read_words method.
        """

        trans = raw_trans.strip().split(',')[0].strip()
        trans = [bipa[x]
                 for part in trans.split(".")
                 for x in tokenizer(part, ipa=True).split()]

        if self.transform is not None:
            trans = [self.transform(s) for s in trans]

        return trans


    def _read_words(self, cog_sets=False):
        """
        Generate the [] of Word entries in the dataset. Raise a DatasetError if
        there is a problem reading the file.

        If the cog_sets flag is set, then yield (Word, cognate class) tuples.
        """
        try:
            with open(self.path, encoding='utf-8', newline='') as f:
                reader = csv.reader(f, dialect=self.dialect)

                header = self._read_header(next(reader),
                        exclude=[] if cog_sets else ['cog_class'])

                self.equilibrium = defaultdict(float)

                for line in reader:
                    asjp = self._read_asjp(line[header['asjp']])

                    for i in asjp:
                        self.equilibrium[i] += 1.0

                    word = Word._make([
                        line[header['doculect']],
                        line[header['concept']],
                        tuple(asjp),
                        None])

                    if cog_sets:
                        yield word, line[header['cog_class']]
                    else:
                        yield word

        except csv.Error as err:
            raise DatasetError('Could not read file: {}'.format(self.path))


    def get_equilibrium(self):
        """
        Return un-normalized equilibrium counts
        """
        if self.equilibrium is None:
            self.get_words()

        return self.equilibrium


    def get_alphabet(self):
        """
        Return a sorted list of all characters found throughout transcriptions
        in the dataset.
        """
        if self.alphabet is not None:
            return self.alphabet

        self.alphabet = set()

        for word in self.get_words():
            self.alphabet |= set(word.sc)

        self.alphabet = sorted(self.alphabet)

        return self.alphabet


    def get_words(self):
        """
        Return the [] of Word named tuples comprising the dataset, excluding
        in-doculect synonyms; i.e. the output should include at most one word
        per doculect per concept.

        Raise a DatasetError if there is an error reading the file.
        """
        words = []
        seen = set()

        for word in self._read_words():
            key = (word.doculect, word.concept,)
            if key not in seen:
                seen.add(key)
                words.append(word)

        return words


    def get_concepts(self):
        """
        Return a {concept: words} dict mapping each concept in the dataset to a
        [] of Word tuples that belong to that concept. In-doculect synonyms are
        excluded.

        Raise a DatasetError if there is an error reading the dataset file.
        """
        d = defaultdict(list)

        for word in self.get_words():
            d[word.concept].append(word)

        return d


    def get_asjp_pairs(self, cutoff=1.0, as_int_tuples=False):
        """
        Return the list of the pairs of transcriptions of words from different
        languages but linked to the same concept.

        If the cutoff arg is less than 1.0, pairs with edit distance above that
        threshold are also ignored. If the other keyword arg is set, return the
        transcriptions as tuples of the letters' indices in self.alphabet.

        Raise a DatasetError if there is an error reading the dataset file.
        """
        pairs = []

        if as_int_tuples:
            alphabet = self.get_alphabet()

        for concept, words in self.get_concepts().items():
            for word1, word2 in itertools.combinations(words, 2):
                if word1.doculect == word2.doculect:
                    continue

                if normalized_levenshtein(word1.sc, word2.sc) > cutoff:
                    continue

                if as_int_tuples:
                    pair = (
                        tuple([alphabet.index(char) for char in word1.sc]),
                        tuple([alphabet.index(char) for char in word2.sc]))
                else:
                    pair = (word1.sc, word2.sc)

                pairs.append(pair)

        return pairs


    def get_clusters(self):
        """
        Return a {concept: cog_sets} dict where the values are frozen sets of
        frozen sets of Word tuples, comprising the set of cognate sets for that
        concept. In-doculect synonyms are excluded.

        Raise a DatasetError if the dataset does not include cognacy info or if
        there is a probelm reading the file.
        """
        d = defaultdict(set)  # {(concept, cog_class): set of words}
        seen = set()  # set of (doculect, concept) tuples
        clusters = defaultdict(list)  # {concept: [frozenset of words, ..]}

        for word, cog_class in self._read_words(cog_sets=True):
            if (word.doculect, word.concept) not in seen:
                seen.add((word.doculect, word.concept))
                d[(word.concept, cog_class)].add(word)

        for (concept, cog_class), cog_set in d.items():
            clusters[concept].append(frozenset(cog_set))

        return {key: frozenset(value) for key, value in clusters.items()}


class CLDFDataset (Dataset):
    """A Dataset subclass for CLDF wordlists. """
    def __init__(self, path, transform=None):
        """Create, based on the path to a CLDF wordlist.

        This constructure assumes that a 'forms.csv' file is a metadata-free
        wordlist, and that any other file is a Wordlist metadata json file.

        Parameters
        ==========
        path: string or Path
            The path to a CLDF wordlist metadata file.
            (Metadata-free wordlists are not supported yet.)
        is_ipa: function Symbol→String or None
            A function to convert bipa Sounds into sound class symbols
            (Use None for no conversion)

        """

        if str(path).endswith("forms.csv"):
            dataset = pycldf.Wordlist.from_data(path)
        else:
            dataset = pycldf.Wordlist.from_metadata(path)

        self.dataset = dataset
        self.transform = transform

        self.alphabet = None

    def _read_words(self, cog_sets=False):
        """
        """
        c_doculect = self.dataset["FormTable", "languageReference"].name
        c_concept = self.dataset["FormTable", "parameterReference"].name
        c_segments = self.dataset["FormTable", "segments"].name
        c_id = self.dataset["FormTable", "id"].name
        if cog_sets:
            try:
                c_cog = self.dataset["FormTable", "cognatesetReference"].name
                lookup = False
            except KeyError:
                c_cog = self.dataset["CognatesetTable", "cognatesetReference"].name
                c_form = self.dataset["CognatesetTable", "formReference"].name
                lookup = {}
                for row in self.dataset["CognatesetTable"].iterdicts():
                    lookup[row[c_form]] = row[c_cog]

        self.equilibrium = defaultdict(float)

        for row in self.dataset["FormTable"].iterdicts():
            if self.transform is None:
                asjp_segments = row[c_segments]
            else:
                asjp_segments = [self.transform(bipa[s]) if bipa[s].name else '0'
                                 for s in row[c_segments]]

            if not asjp_segments:
                continue

            word = Word(
                row[c_doculect],
                row[c_concept],
                tuple(asjp_segments),
                row[c_id])
            for i in asjp_segments:
                self.equilibrium[i] += 1.0

            if cog_sets:
                if lookup:
                    yield word, lookup.get(word.id, None)
                else:
                    yield word, row[c_cog]
            else:
                yield word

class PairsDataset:
    """
    Handles the reading of datasets stored in the training_data dir. These are
    tsv files comprising ASJP word pairs with their respective edit distances.

    Usage:

        try:
            dataset = PairsDataset(path)
            word_pairs = dataset.get_asjp_pairs()
        except DatasetError as err:
            print(err)
    """

    def __init__(self, path, transform=None):
        """
        Set the instance's props. Raise a DatasetError if the given file path
        does not exist.
        """
        if not os.path.exists(path):
            raise DatasetError('Could not find file: {}'.format(path))

        self.path = path
        self.alphabet = None
        self.transform = str if transform is None else transform


    def _read_pairs(self):
        """
        Generate the (asjp, asjp, edit distance) entries from the dataset.
        Raise a DatasetError if there is a problem reading the file.
        """
        with open(self.path, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')

            for row in reader:
                item1 = [self.transform(x)
                         for part in row[0].split(".")
                         for x in tokenizer(part, ipa=True).split()]
                item2 = [self.transform(x)
                         for part in row[1].split(".")
                         for x in tokenizer(part, ipa=True).split()]
                yield item1, item2, float(row[2])


    def get_alphabet(self):
        """
        Return a sorted list of all characters found throughout transcriptions
        in the dataset. Raise a DatasetError if there is a problem.
        """
        if self.alphabet is not None:
            return self.alphabet

        self.alphabet = set()

        for asjp1, asjp2, _ in self._read_pairs():
            self.alphabet |= set(asjp1)
            self.alphabet |= set(asjp2)

        self.alphabet = sorted(self.alphabet)

        return self.alphabet


    def get_asjp_pairs(self, cutoff=1.0, as_int_tuples=False):
        """
        Return the list of the pairs of ASJP transcriptions from the dataset.

        If the cutoff arg is less than 1.0, pairs with edit distance above that
        threshold are also ignored. If the other keyword arg is set, return the
        transcriptions as tuples of the letters' indices in self.alphabet.

        Raise a DatasetError if there is an error reading the dataset file.
        """
        pairs = []

        if as_int_tuples:
            alphabet = self.get_alphabet()

        for asjp1, asjp2, edit_distance in self._read_pairs():
            if edit_distance > cutoff:
                continue

            if as_int_tuples:
                pair = (
                    tuple([alphabet.index(char) for char in asjp1]),
                    tuple([alphabet.index(char) for char in asjp2]))
            else:
                pair = (asjp1, asjp2)

            pairs.append(pair)

        return pairs



def write_clusters(clusters, path=None, dialect='excel-tab'):
    """
    Write cognate set clusters to a csv file with columns: concept, doculect,
    transcription, cog_class. The latter comprises automatically generated id
    strings of the type concept:number.

    The clusters arg should be a dict mapping concepts to frozen sets of frozen
    sets of Word named tuples.

    If path is None, use stdout. Raise a DatasetError if the file/stdout cannot
    be written into.
    """
    if path:
        try:
            f = open(path, 'w', encoding='utf-8', newline='')
        except OSError as err:
            raise DatasetError('Could not open file: {}'.format(path))
    else:
        f = sys.stdout

    writer = csv.writer(f, dialect=dialect)
    writer.writerow(['concept', 'doculect', 'transcription', 'cog_class'])

    for concept, cog_sets in sorted(clusters.items()):
        for index, cog_set in enumerate(cog_sets):
            for word in sorted(cog_set):
                writer.writerow([
                    word.concept, word.doculect, word.sc,
                    '{}:{!s}'.format(concept, index)])

    if path:
        f.close()
