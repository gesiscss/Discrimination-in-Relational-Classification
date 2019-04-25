import sys
import os
from libs.utils.loggerinit import *

class ArgumentsHandler(object):
    FOLDER = '<data>_<label>_T<known>_LC<LC>_RC<RC>_CI<CI>_SEED<seed>_SAMPLING<sampling><RCattributes><LCattributes>'
    FOLDER_STATS = '<data>'
    FILE = 'P<pseeds>_RUN<run>'

    def __init__(self,must):
        self.must = must
        self.arguments = None

    def parse_aguments(self):
        args = sys.argv[1:]
        self.arguments = {}
        for v in args:
            if v.startswith('-') and len(v.strip(' ')) > 1:
                key = v[1:]
            else:
                self.arguments[key] = v
                key = None

        if 'known' not in self.arguments:
            self.arguments['known'] = None

        # import getopt
        #  try:
        #    opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
        # except getopt.GetoptError:
        #    print 'test.py -i <inputfile> -o <outputfile>'
        #    sys.exit(2)
        # for opt, arg in opts:
        #    if opt == '-h':
        #       print 'test.py -i <inputfile> -o <outputfile>'
        #       sys.exit()
        #    elif opt in ("-i", "--ifile"):
        #       inputfile = arg
        #    elif opt in ("-o", "--ofile"):
        #       outputfile = arg

    def are_valid_arguments(self):
        print('Arguments required:')
        print(self.must)
        missing = [int(m not in self.arguments) for m in self.must]
        if sum(missing)>0:
            print('Arguments missing:')
            print('\n'.join(['- {}'.format(self.must[i]) for i,v in enumerate(missing) if v == 1]))
            return False

        evalvalid = ['accuracy','f1','mae','cnf_matrix','error','roc_auc']
        if 'eval' in self.arguments.keys():
            if not self.arguments['eval'] in evalvalid:
                print('{} not in {}'.format(self.arguments['eval'], evalvalid))
        return True

    def get(self,k):
        if k not in self.arguments:
            logging.error('{} argument does not exist.'.format(k))
            sys.exit(0)
        return self.arguments[k]

    def get_file(self):
        fn = str(self.FILE)
        for k, v in self.arguments.items():
            if k == 'pseeds':
                if v != 'all':
                    try:
                        v = int(float(v) * 100)
                    except:
                        logging.error('Truth data only as % of nodes supported [{} given].'.format(v))
                        sys.exit(0)
            fn = fn.replace('<{}>'.format(k), str(v))
        print(fn)
        return fn

    def get_path(self, path=None):

        if path is None or path == self.FOLDER_STATS:
            if path is None:
                path = str(self.FOLDER)
            tmp_data = None
            tmp_known = None
            for k,v in self.arguments.items():
                if k != 'output':
                    if k == 'data':
                        v = v.split('/')[-1].split('.gpickle')[0].split('.edgelist')[0]
                        tmp_data = str(v)

                    elif k == 'known':
                        path = path.replace('<known>', 'random')

                    elif k == 'sampling':
                        ext = '.pickle' if v.endswith('.pickle') else '.gpickle' if v.endswith('.gpickle') else '.txt' if v.endswith('.txt') else ''

                        if ext != '':

                            self.arguments['known'] = str(v)

                            v = v.split('/')[-1].split(ext)[0].split('_') # .pickle since name contains . for B H K

                            if len(v) > 2:
                                tmp_known = '_'.join(v[:-2])
                                v = '{}{}'.format(v[-2].lower(),v[-1].upper())
                            else:
                                tmp_known = str(v[0])
                                v = str(v[1])

                            self.arguments[k] = str(v)

                    elif k == 'RCattributes':
                        if v == 'y':
                            v = '_RCwithAttributes'
                        else:
                            v = ''
                    elif k == 'LCattributes':
                        if v == 'y':
                            v = '_LCwithAttributes'
                        else:
                            v = ''

                    path = path.replace('<{}>'.format(k),str(v))

            if tmp_known is not None and tmp_data is not None:
                if tmp_known.lower() != tmp_data.lower():
                    logging.warning('known:{} might not contain nodes from data:{}'.format(tmp_known,tmp_data))
                    sys.exit(0)

            path = os.path.join(self.arguments['output'], path)

        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except Exception as ex:
                logging.warning(ex)

        print(path)
        return path

    def get_path_stats(self):
        return self.get_path(self.FOLDER_STATS)


