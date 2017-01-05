"""
    fitCurves.py
    Fit curves for LBI data. 
"""
import sys, os
import numpy as np
import argparse
import re
import struct
import codecs
import base64
import collections
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.optimize import least_squares
from scipy.optimize import leastsq
import traceback

class ParseException(Exception):
    pass

class BaseLineException(Exception):
    pass

class ConsistencyException(Exception):
    pass

class ConcentrationException(Exception):
    pass


functions = {'one2one_off'              : (lambda p,x: (p[0]*np.exp(-p[1]*x))),
             'one2one_off_err'          : (lambda p,x,y: (p[0]*np.exp(-p[1]*x) - y)),
             'one2one_on'               : (lambda p,x: (p[0]*(1-np.exp(-p[1]*x)))),
             'one2one_on_err'           : (lambda p,x,y: (p[0]*(1-np.exp(-p[1]*x)) - y)),
             'one2one_off_err_global'   : (lambda p,*args: ( np.concatenate([p[i+1]*np.exp(-p[0]*args[0])-args[i+1] \
                                                                        for i in range(len(p)-1)]))),
             'one2one_on_err_global'    : (lambda p,*args: (np.concatenate([p[i+1]*(1-np.exp(-(p[0]*p[(len(p)-1)/2+1+i] \
                                            + args[0])*args[1])) - args[i+2] for i in range((len(p)-1)/2) ]))),
             'two2one_off'              : (lambda p,x: (p[0]*np.exp(-p[2]*x) + p[1]*np.exp(-p[3]*x))),
             'two2one_off_err'          : (lambda p,x,y: (p[0]*np.exp(-p[2]*x) + p[1]*np.exp(-p[3]*x) -y)),
             'two2one_on'               : (lambda p,x: ( p[0]*(1-np.exp(-p[2]*x)) + p[1]*(1-np.exp(-p[3]*x))      )),
             'two2one_on_err'           : (lambda p,x,y: ( p[0]*(1-np.exp(-p[2]*x)) + p[1]*(1-np.exp(-p[3]*x))  -y   )),
             'two2one_off_err_global'   : (lambda p,*args: (np.concatenate([ p[i+2]*np.exp(-p[0]*args[0])  + \
                                            p[i+(len(p)-2)/2+2]*np.exp(-p[1]*args[0]) - args[i+1] for i in range((len(p)-2)/2)]) )),
             'two2one_on_err_global'    : (lambda p,*args: (   np.concatenate( [p[i+2]*(1-np.exp(-(p[0]*p[2*(len(p)-2)/3+2+i] + \
                                            args[0])*args[2]))  + p[i+(len(p)-2)/3+2]*(1-np.exp(-(p[1]*p[2*(len(p)-2)/3+2+i] + \
                                            args[1])*args[2]))     - args[i+3] for i in range((len(p)-2)/3)])))}               
'''
def two2one_off_err_global(p,*args): 
    n = (len(p)-2)/2        # number of data sets
    kd1 = p[0]           # shared variable
    kd2 = p[1]
    err = []
    for i in range(n):
        err.append( p[i+2]*np.exp(-kd1*args[0])  + p[i+n+2]*np.exp(-kd2*args[0]) - args[i+1] )
    #err = [p[i+1]*np.exp(-kd*args[i]) for i in range(n)]
    return np.concatenate(err)

def two2one_on_err_global(p,*args):
    n = (len(p)-2)/3
    kd1 = args[0]
    kd2 = args[1]
    ka1 = p[0]
    ka2 = p[1]
    err = []
    #kobs = ka*M+kd
    for i in range(n):
        print p[2*n+2+i]
        err.append(p[i+2]*(1-np.exp(-(ka1*p[2*n+2+i] + kd1)*args[2]))  + p[i+n+2]*(1-np.exp(-(ka2*p[2*n+2+i] + kd2)*args[2]))     - args[i+3])
    return np.concatenate(err)

'''
def get_help():
    help = "\n"
    help += "Description: \n"
    help += "    Fit curves for LBI data.\n"
    help += "\n"
    help += "Usage: \n"
    help += "    fitCurves.py --inFiles data1.frd data2.frd ... --outFile outFile --conc_analyte conc1 conc 2... --fit_global=bool \
            --baseline=bool --baseline_file=baseline.frd --model=model\n"
    help += "Input: \n"
    help += "    inputFiles                  LBI output data in .frd format. Exclude baseline file names.              \n"
    help += "    outputDirectory             Ouput directory names. This will be relative to cwd                       \n"
    help += "    conc_analyte                Concentrations of analytes in nM for each input file except baseline.     \n"
    help += "\n"
    help += "Outputs: \n"
    help += "\n"
    help += "Options: \n"
    help += "    --fit_global                    Fit globally? \n"
    help += "    --baseline                  Subtract baseline? \n"
    help += "    --baseline_files            Baseline file names. \n"
    help += "    --model                     Model type to be fit. one2one or two2one\n"
    help += "    --range                     Range to constrain the concentration for global fitting.\n"

    
    return help

def parse_args(input=None):
    parser = argparse.ArgumentParser()
    
    input_group = parser.add_argument_group('Input')
    
    input_group.add_argument('--inFiles', dest='inFiles', nargs='+', type=str, action='store', required=True,
                             help = 'Input filenames with .frd suffix. This will be relative to cwd.', metavar='Filenames')
    
    input_group.add_argument('--outFile', dest='outFile', nargs='?', type=str, action='store', required=True,
                             help = 'Output file name. This will be relative to cwd.', metavar='Filenames')
    
    input_group.add_argument('--conc_analyte', dest='conc_analyte', nargs='+', type=float, action='store', required=True,
                             help = 'Concentrations of analytes in nM for each input file except baseline.', metavar='Parameter')
    
    input_group.add_argument('--fit_global',dest='fit_global', nargs='?', const=1, type=bool, default=False, action='store',required=False,
                             help = 'Fit globally?', metavar='Parameter')
    
    input_group.add_argument('--baseline', dest='baseline', nargs='?', const=1, type=bool, default=False, action='store',required=False,
                             help = 'Subtract baseline?', metavar='Parmeter')
    
    input_group.add_argument('--baseline_files', dest='baseline_files', nargs='?', const=1, type=str, default="", action='store',required=False,
                             help = 'Baseline files.', metavar='Parmeter')
    
    input_group.add_argument('--model', dest='model', nargs='?', const=1, type=str, default="one2one", action='store',required=False,
                             help = 'Select model to be fit.', metavar='Parameter')
    
    input_group.add_argument('--range', dest='range', nargs='?', const=1, type=float, default=50, action='store',required=False,
                             help = 'Range to constrain the concentration for global fitting.', metavar='Parameter')
    if input:
        return parser.parse_args(input)
    else:
        return parser.parse_args()
 
def parse_frd(filename,stepName):
    try:
        file = codecs.open(filename,'r',encoding='utf8')             # read in file as binary data. 
        data = file.read()
        stepNames = re.findall('<StepName>(.*?)<',data,re.DOTALL)
        x_points = re.findall('AssayXData Points="(.*?)">',data,re.DOTALL)
        y_points = re.findall('AssayYData Points="(.*?)">',data,re.DOTALL)
        
        x_data_uncoded = [sub[1] for sub in re.findall('AssayXData Points="(.*?)">(.*?)</AssayXData',data,re.DOTALL)]
        x_data_decoded = [''.join([base64.b64decode(d) for d in dat.split()]) for dat in x_data_uncoded]
        x_data = [struct.unpack('%if' % (len(s)/4),s) for s in x_data_decoded]
        y_data_uncoded = [sub[1] for sub in re.findall('AssayYData Points="(.*?)">(.*?)</AssayYData',data,re.DOTALL)]
        y_data_decoded = [''.join([base64.b64decode(d) for d in dat.split()]) for dat in y_data_uncoded]
        y_data = [struct.unpack('%if' % (len(s)/4),s) for s in y_data_decoded]
        
        data = {filename:collections.OrderedDict([(stepNames[i],{'x_data':x_data[i],'y_data':y_data[i]}) for i in \
                            range(len(stepNames)) if (stepName in stepNames[i])])} # keep only requested stepName
        
        
        # offset x_data to start at t=0
        for step_key, step_value in data.values()[0].items():
            for axis_key, axis_value in step_value.items():
                if 'x' in axis_key:
                    offset = axis_value[0]          # 1st element 
                    data.values()[0][step_key][axis_key] = [s - offset for s in data.values()[0][step_key][axis_key]]
                else:
                    continue
        return data

    except Exception:
        print traceback.format_exc()
        raise ParseException()

def check_size(data, args, data_baseline = {}):
    #This function checks for consistency of step names and number of points in each step for each data set. 
    num_points = {stepName:len(data.values()[0][stepName].values()[0]) for stepName in data.values()[0].keys()} # get numbers from the first data set
    consistent = True
    for dset_key, dset_value in data.items():
        for step_key, step_value in dset_value.items():
            for axis_key, axis_value in step_value.items():
                if len(axis_value) != num_points[step_key]:         
                    consistent = False
                if data_baseline:
                    for bset_key, bset_value in data_baseline.items():
                        if len(bset_value[step_key][axis_key]) != num_points[step_key]:
                            consistent = False
    return consistent

def fit_one2one(data, args):
    
    f1, ax = plt.subplots(1,1)
    n_dat = len(data)
    
    if args.fit_global:
        Aoff,Aon,kobs = ([None]*n_dat for i in range(3))

        x_on = np.array(data.values()[0]['Association']['x_data'])      # assuming that all data sets have the same x values
        x_off = np.array(data.values()[0]['Dissociation']['x_data'])  
        
        y_on,y_off = (tuple((np.array(data[key]['Association']['y_data']) for key in data.keys())),        # for each data set get y values
                            tuple((np.array(data[key]['Dissociation']['y_data']) for key in data.keys())))
        # non-linear least squares optimization. Unconstrained 
        p_best_off = least_squares(functions['one2one_off_err_global'], [.1 for i in range(n_dat+1)] ,args=((x_off,)+y_off))
        kd, Aoff = (p_best_off.x[0], p_best_off.x[1:])
        
        # non-linear least squares optimization. Constrain kon>kd, |conc| < args.conc_analyte +/- args.range 
        if args.range:
            constraints = tuple([[kd]+[-np.inf]*n_dat+[conc-args.range for conc in args.conc_analyte], 
                             [np.inf]+[np.inf]*n_dat+[conc+args.range for conc in args.conc_analyte]    ] )
        else: 
            constraints = tuple([[kd]+[-np.inf]*n_dat+[-np.inf for conc in args.conc_analyte], 
                             [np.inf]+[np.inf]*n_dat+[np.inf for conc in args.conc_analyte]    ] )
        
        try:
            p_best_on = least_squares(functions['one2one_on_err_global'], [.1 for i in range(2*n_dat+1)], 
                                  args=((np.array([kd for i in range(len(x_on))]),) + (x_on,)  + y_on), bounds=constraints) 
        except ValueError:
            print 'Infeasible constraints\n.'
            sys.exit()
       
        kon = p_best_on.x[0]
        Aon = p_best_on.x[1:n_dat+1] 
        kobs = p_best_on.x[n_dat+1:]*kon+kd
        
        kD = kd/kon
        for i in range(n_dat):
            
            offset= x_on[-1]      # Since we shifted the data set for both on and off to start at 0, we need to shift the off relative to on before plotting. 
            delta = x_on[-2]-x_on[-1]   # This is to account for the spacing between data points. 
            ax.plot(np.append(x_on,x_off+offset+delta),np.append(y_on[i],y_off[i]))   # plot the original data
            y_pred_on = functions['one2one_on']([Aon[i],kobs[i]],x_on)          # calculate predicted values
            y_pred_off = functions['one2one_off']([Aoff[i],kd],x_off)
            
            ax.plot(np.append(x_on,x_off+offset+delta),np.append(y_pred_on,y_pred_off))
        kD = kd/kon
        outName = args.outFile.split('.')[0]
        f1.savefig(outName + '.pdf')
        outfile = open(outName +'.txt','w')
        outfile.write('Model: %10s\n' % args.model )
        outfile.write('Global: %10r\n' % args.fit_global)
        outfile.write('Estimated koff: %5f s-1\n ' %(kd))
        outfile.write('Estimated kon: %5f M-1 s-1\n ' %(kon))
        outfile.write('Estimated kD: %5f M-1\n' %(kD))
        
    else:
        
        Aoff,Aon,kd,kobs,ka,kD,ieroff,ieron = ([None]*len(data.keys()) for i in range(8))
        for i, key in enumerate(data.keys()):     # Have to cast list into numpy array to use leastsq function. Change this at data parsing stage?
            x_on,y_on,x_off,y_off = (np.array(data[key]['Association']['x_data']), np.array(data[key]['Association']['y_data']),
                                                    np.array(data[key]['Dissociation']['x_data']),np.array(data[key]['Dissociation']['y_data']))
            
            p_best_off = least_squares(functions['one2one_off_err'],[1,.1],args=(x_off,y_off))
            Aoff[i], kd[i] = (p_best_off.x[0], p_best_off.x[1])
            
            #(Aoff[i], kd[i]), ieroff[i] = least_squares(functions['one2one_off_err'],[1,.1],args=(x_off,y_off))
            constraints = tuple([[-np.inf]+[kd[i]], [np.inf]+[np.inf]    ] )
            try:
                p_best_on =  least_squares(functions['one2one_on_err'],[1,.1],args=(x_on,y_on), bounds=constraints)
                Aon[i], kobs[i] = (p_best_on.x[0], p_best_on.x[1])
            except ValueError:
                print 'Infeasible constraints\n.'
                sys.exit()
            #(Aon[i], kobs[i]), ieron[i] = leastsq(functions['one2one_on_err'],[1,.1],args=(x_on,y_on))
            ka[i] = (kobs[i]-kd[i])/args.conc_analyte[i]
            kD[i] = kd[i]/ka[i]
            
            offset= x_on[-1]      # Since we shifted the data set for both on and off to start at 0, we need to shift the off relative to on before plotting. 
            delta = x_on[-2]-x_on[-1]   # This is to account for the spacing between data points. 
            ax.plot(np.append(x_on,x_off+offset+delta),np.append(y_on,y_off))   # plot the original data
            
            y_pred_on = functions['one2one_on']([Aon[i],kobs[i]],x_on)          # calculate predicted values
            y_pred_off = functions['one2one_off']([Aoff[i],kd[i]],x_off)

            ax.plot(np.append(x_on,x_off+offset+delta),np.append(y_pred_on,y_pred_off))
        
        # save figure, output optimized parameters

        outName = args.outFile.split('.')[0]
        f1.savefig(outName + '.pdf')
        outfile = open(outName +'.txt','w')
        outfile.write('Model: %10s\n' % args.model )
        outfile.write('Global: %10r\n' % args.fit_global)
        outfile.write('Estimated koff:\n ' + '%5f s-1\n'*len(kd) %(tuple(kd)))
        outfile.write('Estimated kon:\n '+ '%5f M-1 s-1\n'*len(ka) %(tuple(ka)))
        outfile.write('Estimated kD:\n '+ '%5f M-1\n'*len(kD) %(tuple(kD)))
    plt.show()
    
def fit_two2one(data, args):
    
    f1, ax = plt.subplots(1,1)
    n_dat = len(data)

    if args.fit_global:
        Aoff,Aon,kobs = ([None]*n_dat for i in range(3))

        x_on = np.array(data.values()[0]['Association']['x_data'])      # assuming that all data sets have the same x values
        x_off = np.array(data.values()[0]['Dissociation']['x_data'])  
        
        y_on,y_off = (tuple((np.array(data[key]['Association']['y_data']) for key in data.keys())),        # for each data set get y values
                            tuple((np.array(data[key]['Dissociation']['y_data']) for key in data.keys())))
        
        constraints = tuple([ [0]+[0]+[0]*n_dat*2,[np.inf]+[np.inf]+[np.inf]*n_dat*2    ])
        
        try:
            p_best_off = least_squares(functions['two2one_off_err_global'], [.1 for i in range(2*n_dat+2)] ,args=((x_off,)+y_off),bounds=constraints)
        except ValueError:
            print 'Infeasible constraints.\n'
            sys.exit()
            
        kd= (p_best_off.x[0],p_best_off.x[1])
        Aoff = zip(*zip(*[iter(p_best_off.x[2:])]*n_dat))

        if args.range:
            constraints = tuple([[0]+[0]+[0]*n_dat*2+[conc-args.range for conc in args.conc_analyte], 
                             [np.inf]+[np.inf]+[np.inf]*n_dat*2+[conc+args.range for conc in args.conc_analyte]] )

        else: 
            constraints = tuple([[0]+[0]+[0]*n_dat*2+[0 for conc in args.conc_analyte], 
                             [np.inf]+[np.inf]+[np.inf]*n_dat+[np.inf for conc in args.conc_analyte]] )
        
        try:
            p_best_on = least_squares(functions['two2one_on_err_global'], [.1 for i in range(2*n_dat+2)]+[ c for c in args.conc_analyte], \
                                 args=( (np.array([kd[0] for i in range(len(x_on))]),)+ (np.array([kd[1] for i in range(len(x_on))]),) +(x_on,)+y_on) ,bounds=constraints)  
        except ValueError:
            print 'Infeasible constraints.\n'
            sys.exit()
        #p_best, ierron = leastsq(functions['two2one_on_err_global'], [.1 for i in range(2*n_dat+2)]+[ c*10**-9 for c in args.conc_analyte], \
                                 #args=( (np.array([kd[0] for i in range(len(x_on))]),)+ (np.array([kd[1] for i in range(len(x_on))]),) +(x_on,)+y_on) )
        kon = p_best_on.x[:2]
        Aon =  zip(*zip(*[iter(p_best_on.x[2:n_dat*2+2])]*n_dat))
        conc = p_best_on.x[n_dat*2+2:]   
        #kobs = zip(*zip(*[iter(p_best[n_dat*2+2:])]*n_dat))
        kobs = [(conc[i]*kon[0]+kd[0], conc[i]*kon[1]+kd[1]) for i in range(len(conc))]
        
        for i in range(n_dat):
            
            offset= x_on[-1]      # Since we shifted the data set for both on and off to start at 0, we need to shift the off relative to on before plotting. 
            delta = x_on[-2]-x_on[-1]   # This is to account for the spacing between data points. 
            ax.plot(np.append(x_on,x_off+offset+delta),np.append(y_on[i],y_off[i]))   # plot the original data
           
            y_pred_on = functions['two2one_on'](Aon[i]+kobs[i],x_on)          # calculate predicted values
            y_pred_off = functions['two2one_off'](Aoff[i]+kd,x_off)
        
            #ax.plot(x_off+offset+delta,y_pred_off)
            ax.plot(np.append(x_on,x_off+offset+delta),np.append(y_pred_on,y_pred_off))
        
        # output data to file and save figure

        outName = args.outFile.split('.')[0]
        f1.savefig(outName + '.pdf')
        outfile = open(outName +'.txt','w')
        outfile.write('Model: %10s\n' % args.model )
        outfile.write('Global: %10r\n' % args.fit_global)
        outfile.write('Estimated koff: %5f s-1 %5f s-1\n' %(kd))

        #outfile.write('Estimated Aoff: '+ '%5f %5f '*len(Aoff) +' \n' %(tuple([ i for sub in Aoff for i in sub] )))
        outfile.write('Estimated kon: %5f M-1s-1 %5f M-1s-1\n' %(tuple(kon)))
        #outfile.write('Estimated Aon: %5f %5f \n' %(Aon))
        outfile.write('Estimated kD: %5f M-1 %5fM-1\n' %(tuple([i/j for i,j in zip(kd,tuple(kon))  ])))
        #outfile.write('Estimated conc:'+ '%5f %5f'*len(conc) %(tuple( )))

    else:
        Aoff,Aon,kd,kobs,ka,Kd,ieroff,ieron = ([None]*len(data.keys()) for i in range(8))
        for i, key in enumerate(data.keys()):     # Have to cast list into numpy array to use leastsq function. Change this at data parsing stage?
            x_on,y_on,x_off,y_off = (np.array(data[key]['Association']['x_data']), np.array(data[key]['Association']['y_data']),
                                                    np.array(data[key]['Dissociation']['x_data']),np.array(data[key]['Dissociation']['y_data']))

            constraints = tuple([[0]+[0]+[0]+[0],[np.inf]+[np.inf]+[np.inf]+[np.inf]  ])
    
            try:
                p_best_off = least_squares(functions['two2one_off_err'],[1,1,.1,.1],args=(x_off,y_off),bounds=constraints)
            except ValueError:
                print 'Infeasible constraints.\n'
                sys.exit()            
                
            Aoff[i] = tuple(p_best_off.x[:2])
            kd[i] = tuple(p_best_off.x[2:])
            
            constraints = tuple([[0]+[0]+[kd[i][0]]+[kd[i][1]],[np.inf]+[np.inf]+[np.inf]+[np.inf]] )

            try:
                p_best_on = least_squares(functions['two2one_on_err'],[1,1,.1,.1],args=(x_on,y_on),bounds=constraints)
            except ValueError:
                print 'Infeasible constraints.\n'
                sys.exit()   
            Aon[i] = tuple(p_best_on.x[:2])
            kobs[i] = tuple(p_best_on.x[2:])
            ka[i] = ((kobs[i][0] - kd[i][0])/args.conc_analyte[i],  (kobs[i][1] - kd[i][1])/args.conc_analyte[i])
            
            
            offset= x_on[-1]      # Since we shifted the data set for both on and off to start at 0, we need to shift the off relative to on before plotting. 
            delta = x_on[-2]-x_on[-1]   # This is to account for the spacing between data points. 
            ax.plot(np.append(x_on,x_off+offset+delta),np.append(y_on,y_off))   # plot the original data
                        
            y_pred_on = functions['two2one_on'](Aon[i]+kobs[i],x_on)          # calculate predicted values
            y_pred_off = functions['two2one_off'](Aoff[i]+kd[i],x_off)
            ax.plot(np.append(x_on,x_off+offset+delta), np.append(y_pred_on, y_pred_off))

        kD = [(kd[i][0]/ka[i][0],kd[i][1]/ka[i][1]) for i in range(len(kd))]
        
        outName = args.outFile.split('.')[0]
        f1.savefig(outName + '.pdf')
        outfile = open(outName +'.txt','w')
        outfile.write('Model: %10s\n' % args.model )
        outfile.write('Global: %10r\n' % args.fit_global)
        outfile.write('Estimated koff:\n ' +'%5f s-1 %5f s-1\n'*len(kd)  %(tuple([i for sub in kd for i in sub]  )))
        outfile.write('Estimated kon: \n' + ' %5f M-1s-1 %5f M-1s-1\n'*len(ka)  %(tuple([i for sub in ka for i in sub] )))
        outfile.write('Estimated kD: \n' + ' %5f M-1 %5f M-1\n'*len(kD)  %(tuple([i for sub in kD for i in sub] ))  )
        #outfile.write('Estimated kD: %5f M-1 %5fM-1\n' %(tuple([i/j for i,j in zip(kd,tuple(kon))  ])))
    
    plt.show()
    
def fit(args):
    
    try:
        # Parse data, check for consistency, subtract baseline. 
        
        try:
            if type(args.inFiles) == list:
                
                data = collections.OrderedDict([(inFile,parse_frd(inFile,'ation')[inFile]) for inFile in args.inFiles])
                data_baseline2 = collections.OrderedDict([(inFile,parse_frd(inFile,'Baseline2')[inFile]) for inFile in args.inFiles])
            else:
                data = {args.inFiles:parse_frd(args.inFiles,'ation')[args.inFiles]}   
                data_baseline2 = {args.inFiles:parse_frd(args.inFiles,'Baseline2')[args.inFiles]}   
        except ParseException:
            print "Could not parse input files."
            sys.exit()
            
        if args.baseline:
            try:
                if not args.baseline_files:
                    raise BaselineException()
                if type(args.baseline_files) == list:
                    data_baseline = {basefile:parse_frd(basefile,'ation')[basefile] for basefile in args.baseline_files}
                    baseline_baseline2 = {basefile:parse_frd(basefile,'Baseline2')[basefile] for basefile in args.baseline_files}
                else:
                    data_baseline = {args.baseline_files:parse_frd(args.baseline_files,'ation')[args.baseline_files]}
                    baseline_baseline2 = {args.baseline_files:parse_frd(args.baseline_files,'Baseline2')[args.baseline_files]}
            
            except BaseLineException:
                print "Please set baseline file name flag as --baseline_files=filename.frd"
                sys.exit()
            
            except ParseException:
                print "Could not parse baseline files."
                print sys.exit()
        try:
            if args.baseline:
                if not check_size(data, args, data_baseline): # Check to see all data sets have the same number of elemets. 
                    raise ConsistencyException()
            else:
                if not check_size(data, args):
                    raise ConsistencyException()
            
        except ConsistencyException:
            print "Data points are not consistent. All data sets have to have the same number of elements."
            sys.exit()
            
        # align data based on Baseline2 traces
        if args.baseline:
            baseline2 = [np.array(data_baseline2[key]['Baseline2']['y_data']) for key in data_baseline2.keys()]
            baseline2 += [np.array(baseline_baseline2[key]['Baseline2']['y_data']) for key in baseline_baseline2.keys()]
        else:
            baseline2 = [np.array(data_baseline2[key]['Baseline2']['y_data']) for key in data_baseline2.keys()]
        
        c = [np.sum([j-k for j,k in zip(baseline2[0],base) ])/len(baseline2[0]) for base in baseline2]       # c = argmin [sigma(f(x)-g(x)-c)]^2
        
        # add c to each y data point
        if args.baseline:
            for dset_key, dset_value in data_baseline.items():
                for step_key, step_value in dset_value.items():
                    for axis_key, axis_value in step_value.items():
                        if 'y' in axis_key:
                            for bset_key, bset_value in data_baseline.items():
                                data_baseline[dset_key][step_key][axis_key] = [dat+c[-1] for dat in data_baseline[dset_key][step_key][axis_key]]
                    else:
                        continue
                    
            # subtract baseline y from data set
            for dset_key, dset_value in data.items():
                for step_key, step_value in dset_value.items():
                    for axis_key, axis_value in step_value.items():
                        if 'y' in axis_key:
                            for bset_key, bset_value in data_baseline.items():
                                data[dset_key][step_key][axis_key] = [d-b for d,b in zip(data[dset_key][step_key][axis_key], \
                                                    data_baseline[bset_key][step_key][axis_key])]#bset_value[step_key][axis_key])]
                        else:
                            continue
        
        else:            
            counter = 0
            for dset_key, dset_value in data.items():
                for step_key, step_value in dset_value.items():
                    for axis_key, axis_value in step_value.items():
                        if 'y' in axis_key:
                            for bset_key, bset_value in data_baseline.items():
                                data[dset_key][step_key][axis_key] =[dat - c[counter] for dat in data[dset_key][step_key][axis_key]]
                                counter += 1
                        else:
                            continue
       
        try:
            if not len(args.conc_analyte) == len(data.keys()):
                raise ConcentrationException()
        
        except ConcentrationException:
            print "Please set concentration flag as --conc_analyte conc1 conc2 conc3 ... for each input file, excluding baseline."       
        

        
        # Everything checks out so far, so on to fitting curve. 
        
        if args.model == 'one2one':
            fit_one2one(data,args)
            
        elif args.model == 'two2one':
            fit_two2one(data,args)
        else:
            print "Please enter a correct model name. one2one or two2one"
        
                        
    except Exception:
        print "Somthing went wrong..."
        print traceback.format_exc() 
        sys.exit()
        
def leastsq_bounds( func, x0, bounds, boundsweight=10, **kwargs ):
    # this function is borrowed and modified from http://stackoverflow.com/questions/9878558/scipy-optimize-leastsq-with-bound-constraints
    pass

        

if __name__=='__main__':
    
    sys.path.insert(1,os.getcwd())                      # add current directory to pythonpath
    
    if len(sys.argv) < 2 or '-h' in sys.argv[1:] or '--help' in sys.argv[1:] :
        print get_help()
        sys.exit()
        

    elif len(sys.argv[1:])==1:
        file = open(sys.argv[1])
        inArgs = file.read()
        print inArgs
        args = parse_args(inArgs.split())
    else:
        args = parse_args()
    args.conc_analyte
    fit(args)
    
        
        
        
        
        
        
        
        
        
