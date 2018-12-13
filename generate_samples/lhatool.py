#pylint: disable=C0303,C0103


from collections import OrderedDict
import re

DEBUG = False

class LHA(object):
    """docstring for LHA."""
    def __init__(self, infile):
        super(LHA, self).__init__()
        
        self.blocks = OrderedDict()
        self.decays = OrderedDict()
        
        
        if type(infile) == file:
            self.process_input(infile)
        elif type(infile) == str:
            with open(infile) as fin:
                self.process_input(fin)
        else:
            raise Exception('Invalid input file')
        
        
    def get_block(self, name):
        blk = self.blocks[name]
        if not blk:
            raise Exception('Block not found: %s' % name)
        return blk
    
    def get_decay(self, pid):
        dec = self.decays[pid]
        if not dec:
            raise Exception('Decay not found: PDG %d' % pid)
        return dec
        
    
    def add_block(self, block):
        self.blocks[block.title] = block
    
    def add_decay(self, decay):
        self.decays[decay.pdgid] = decay
    
    
    def linetype(self, line):
        
        type_regexes = {'BLOCK': r'(?i)^(\bBLOCK\b)',
                        'DECAY': r'(?i)^(\bDECAY\b)',
                        'ENTRY':  r'^\s*(\d|[\+\-]\d)',
                        'COMMENT' : r'^\s*(\#)'}
        
        thistype = 'UNKNOWN'
        for typ, regxs in type_regexes.iteritems():
            if re.findall(regxs, line):
                thistype = typ
        
        return thistype
        
        
        
    def entrytype(self, string):
        
        type_regexes = {'INT': r'^([+\-]?\d+)(?![\d^\D])',
                        'FLOAT': r'(?:[\+\-])?\d+\.\d+(?:[eEdD][\+\-]\d+)?(?!\.)',
                        'STRING': r'([a-zA-Z]\S*)',
                        'COMMENT' : r'(?:\#.*)'}
        
        thistype = 'UNKNOWN'
        for typ, regxs in type_regexes.iteritems():
            if re.findall(regxs, string):
                thistype = typ
        
        return thistype
    
    
    
    def process_input(self, lhafile):
        
        
        
        lhainput = lhafile.readlines()
        
        current_decay_or_block = None
        
        for i, line in enumerate(lhainput):
            
            ltype = self.linetype(line)
            
            # Fill a new block
            if ltype == 'BLOCK':
                
                blockname = re.search(r'(?i)(?<=block)\s*(\S*)', line)
                if blockname:
                    blockname = blockname.group(1)
                else:
                    raise Exception('Unnamed block at line %d' % (i+1))
                    
                comment = re.search(r'(?i)(?<=\#)\s(.*)', line)
                if comment:
                    comment = comment.group()
                
                self.blocks[blockname] = Block(blockname, comment)
                current_decay_or_block = self.blocks[blockname]
                
                if DEBUG:
                    print 'Found block:', blockname, comment
            
            # Fill entries to the current block
            elif ltype == 'ENTRY':
                
                if current_decay_or_block is None:
                    raise Exception('Entry w/o block at line %d' % (i+1))            
        
                splitline = line.split()
                vals = []
                comm = None
                
                for element in splitline:
                    etype = self.entrytype(element)
                    if etype == 'FLOAT':
                        try:
                            vals.append(float(element))
                        except ValueError:
                            vals.append(str(element))
                    elif etype == 'INT':
                        vals.append(int(element))
                    elif etype == 'STRING' or etype == 'VERSION':
                        vals.append(str(element))
                    elif etype == 'COMMENT':
                        # Merge all remainders of line, and stop
                        comm = line[line.find('#'):]
                        break
                
                entry = Entry(vals, comment=comm)
                
                
                if DEBUG:
                    print 'Processed entry:', entry
                
                current_decay_or_block.entries.append(entry)
                
            
            # Fill decay block
            elif ltype == 'DECAY':
                
                params = re.search(r'((?:[\+\-])?\d+(?!\.))\s*([+\-]?\d?\.?\d*[Ee][+\-]?\d+)', line)
                if params:
                    pid = int(params.group(1))
                    pwidth = params.group(2)
                else:
                    raise Exception('Invalid DECAY block at line %d' % i)
                
                comment = re.search(r'(?i)(?<=\#)\s(.*)', line)
                if comment:
                    comment = comment.group()
                
                
                self.decays[pid] = Decay(pid, pwidth, comment)
                current_decay_or_block = self.decays[pid]
                
                if DEBUG:
                    print 'Found decay:', pid, pwidth, comment
            
            
            # Fill a comment
            elif ltype == 'COMMENT':
                
                # Keep only comments belonging to a block
                if current_decay_or_block is not None:
                    
                    entry = Entry(comment=line)
                    current_decay_or_block.entries.append(entry)
                    
                    if DEBUG:
                        print 'Processed comment:', entry
                
                
    def write(self, outfile):
        
        fout = outfile
        if type(fout) == file:
            keep_open = True
        elif type(fout) == str:
            keep_open = False
            fout = open(fout, 'w')
        
        # Write blocks
        for key, block in self.blocks.iteritems():
            fout.write(block.__str__() + '\n')
            for entry in block.entries:
                fout.write(entry.__str__() + '\n')
        
        # Write decays
        for pid, decay in self.decays.iteritems():
            fout.write(decay.__str__() + '\n')
            for entry in decay.entries:
                fout.write(entry.__str__() + '\n')
        
        if not keep_open:
            fout.close()
        
    
class Block(object):
    """docstring for Block."""
    def __init__(self, title, comment=None):
        super(Block, self).__init__()
        
        self.title = title
        self.comment = comment
        self.entries = []
    
    def __str__(self):
        """ Stringification """
        s = 'BLOCK\t{}'.format(self.title)
        if self.comment is not None:
            s += '\t# {}'.format(self.comment)
        return s
    
    def add(self, entry):
        self.entries.append(entry)
        
        
    def get_entry_by_key(self, key):
        for entry in self.entries:
            if key in entry.lookup:
                return entry.lookup[key]
        raise Exception('No entry with key %s' % key)
    
    def get_entry_by_index(self, index):
        if index >= len(self.entries):
            raise Exception('Requested index (%d) >= number of entries (%d)'
                            % (index, len(self.entries)))



class Entry(object):
    """docstring for Entry."""
    def __init__(self, values=None, comment=None):
        super(Entry, self).__init__()
        
        # Values = None -> comment only
        if values is not None:
            if not isinstance(values, list):
                values = list(values)
        
        self.values = values if values is not None else list()
        self.comment = comment
        self.lookup = {}
        
        # If we have only two values and the first is an int, create a 
        # dictionary for easy lookup
        if len(self.values) == 2:
            self.lookup[self.values[0]] = self.values[1]
            
            if DEBUG:
                print 'Created key-value pair ({}, {}})'.format(self.values[0], self.values[1])
            
        
        
    def __str__(self):
        """ Stringification """
        
        s = ''
        
        for v in self.values:
            if type(v) == int:
                s += '\t{:d}'.format(v)
            elif type(v) == float:
                s += '\t{:.8e}'.format(v)
            else:
                s += '\t{}'.format(v)
        
        if self.comment is not None:
            if s:
                s += '\t'
            s += '{}'.format(self.comment.replace('\n', ''))
            
        return s
    
    def __repr__(self):
        return self.__str__()



        
class Decay(object):
    """docstring for Decay."""
    def __init__(self, pdgid, width, comment=None):
        super(Decay, self).__init__()
        
        self.pdgid = int(pdgid)
        self.width = float(width)
        self.comment = comment
        self.entries = []
    
    
    def get_branching_ratio(self, id1, id2):
        for entry in self.entries:
            if len(entry.values) >= 4:
                if entry.values[2] == int(id1) and entry.values[3] == int(id2):
                    return entry.values[0] 
        
        raise Exception('No BR for pids {}, {}'.format(id1, id2))
    
    def __str__(self):
        """ Stringification """
        s = 'DECAY\t{:d}\t{:.8e}\t# {}'.format(self.pdgid, self.width, self.comment)
        if self.comment is not None:
            s += '\t{}'.format(self.comment.replace('\n', ''))
        
        return s
    
    def __repr__(self):
        return self.__str__()



## -----------------------------------------------------------------------------
if __name__ == '__main__':
    
    # Open a LHA file
    lha = LHA('test.lha')
    
    # Get value
    mA = lha.get_block('MASS').get_entry_by_key(36)
    print 'mA =', mA
    
    wA = lha.get_decay(36).width
    print 'wA =', wA
    
    # Modify existing value
    # TODO
    
    # Add a block
    newblock = Block('TESTBLOCK', comment='Just for testing')
    newentry = Entry(values=[1, 1.2345])
    newblock.add(newentry)
    lha.add_block(newblock)
    
    # Write out a new file
    lha.write('outtest.lha')
    
        