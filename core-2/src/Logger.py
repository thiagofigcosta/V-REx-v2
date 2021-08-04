#!/bin/python
# -*- coding: utf-8 -*-

import sys
import os
from datetime import datetime
import socket
import traceback
try{
    from cStringIO import StringIO # Python 2
}except ImportError{
    from io import StringIO # Python 3
}

class Logger(object){
    DATETIME_FORMAT='%d/%m/%Y-%H:%M:%S'
    DATE_FORMAT='%Y%m%d'
    EYE_CATCHER=False

	def __init__(self, log_folder,verbose=False,print_on_screen=True,name='application'){
		if not log_folder.endswith('\\' if os.name == 'nt' else '/'){
            log_folder+='\\' if os.name == 'nt' else '/'
        }
		self.log_folder=log_folder
        self.print_on_screen=print_on_screen
		self.print_verbose=verbose
		self.name=name
    }
		
	def _log(self,message,error=False,traceback=False,warn=False,fatal=False,clean=False){
		if not clean{
			now = datetime.now()
			nowstr='{}.{:06d}'.format(now.strftime(Logger.DATETIME_FORMAT),now.microsecond)
			info_header="[{}|{}] ".format(socket.gethostname(),nowstr)
			if fatal {
				info_header+='- FATAL: '
			}elif error {
				info_header+='- ERROR: '
			}elif traceback {
				info_header+='- EXCEPTION: '
			}elif warn {
				info_header+='- WARN: '
			}else{
				info_header+='- INFO: '
			}
		}else{
			info_header=''
		}
		fail_delimiter="***********************************************"
		error_header  ="*--------------------ERROR--------------------*"
		traceb_header ="*------------------TRACE_BACK------------------"
		formatted_message =""
		if (error or traceback) and Logger.EYE_CATCHER{
			formatted_message='{}\n{}\n'.format(info_header,fail_delimiter)
			if error{
				formatted_message+='{}\n'.format(error_header)
            }
			if traceback{
				formatted_message+='{}\n'.format(traceb_header)
            }
			formatted_message+='{}\n'.format(fail_delimiter)
			formatted_message+='{}\n'.format(message)
			formatted_message+=fail_delimiter
		}else{
			formatted_message=info_header+message
        }
		if self.print_on_screen{
			if error{
				sys.stderr.write(formatted_message+'\n')
				sys.stderr.flush()
			}else{
				print (formatted_message, flush=True)
            }
        }
		with open(self.getLogFilename(), 'a') as logfile{
			logfile.write(formatted_message+'\n')
        }
    }

    def _handleException(self,e){
		exc_type, exc_value, exc_traceback = sys.exc_info()
		exceptionstr='\n*** message: {}\n'.format(e)
		if exc_traceback is not None{
			fname = os.path.split(exc_traceback.tb_frame.f_code.co_filename)[1]
			exceptionstr="*** file_name: "
			exceptionstr+=fname+'\n'
			exceptionstr+="*** exception:\n"
			exceptionstr+=str(e)+"\n"
			exceptionstr+="*** print_tb:\n"
		}
		str_io = StringIO()
		traceback.print_tb(exc_traceback, limit=1, file=str_io)
		exceptionstr+=str_io.getvalue()
		exceptionstr+="*** print_exception:\n"
		str_io = StringIO()
		traceback.print_exception(exc_type, exc_value, exc_traceback,limit=2, file=str_io)
		exceptionstr+='\t'+str_io.getvalue()
		exceptionstr+="*** print_exc:\n"
		str_io = StringIO()
		traceback.print_exc(limit=2, file=str_io)
		exceptionstr+='\t'+str_io.getvalue()
		exceptionstr+="*** format_exc, first and last line:\n"
		formatted_lines = traceback.format_exc().splitlines()
		exceptionstr+='\t'+formatted_lines[0]+"\n"
		exceptionstr+='\t'+formatted_lines[-1]+"\n"
		format_exception=traceback.format_exception(exc_type, exc_value,exc_traceback)
		if len(format_exception)>0{
			exceptionstr+="*** format_exception:\n"
			for el in format_exception{
				str_el=str(el)
				if not str_el.startswith('\t'){
					str_el='\t'+str_el
				}
				exceptionstr+='{}\n'.format(el)
			}
		}
		extract_tb=traceback.extract_tb(exc_traceback)
		if len(extract_tb)>0{
			exceptionstr+="*** extract_tb:\n"
			for el in extract_tb{
				str_el=str(el)
				if not str_el.startswith('\t'){
					str_el='\t'+str_el
				}
				exceptionstr+='{}\n'.format(el)
			}
		}
		format_tb=traceback.format_tb(exc_traceback)
		if len(format_tb)>0{
			exceptionstr+="*** format_tb:\n"
			for el in format_tb{
				str_el=str(el)
				if not str_el.startswith('\t'){
					str_el='\t'+str_el
				}
				exceptionstr+='{}\n'.format(el)
			}
		}
		if exc_traceback is not None{
			exceptionstr+="*** At line: "
			exceptionstr+=str(exc_traceback.tb_lineno)
		}
		return exceptionstr
    }

	def getLogFilename(self){
		now = datetime.now()
		return self.log_folder+'log-{}-{}.txt'.format(self.name,now.strftime(Logger.DATE_FORMAT))
	}

    def info(self,message){
        self._log(message,error=False,traceback=False,warn=False)
    }

    def fatal(self,message){
        self.error(message,fatal=True)
    }

    def error(self,message,fatal=False){
        self._log(message,error=True,traceback=False,warn=False,fatal=fatal)
        if fatal{
            exit(1)
        }
    }

    def exception(self,e,fatal=False){
        self._log(self._handleException(e),error=False,traceback=True,warn=False,fatal=fatal)
        if fatal{
            exit(1)
        }
    }

    def warn(self,message){
        self._log(message,error=False,traceback=False,warn=True)
    }

	def clean(self,message){
        self._log(message,clean=True)
    }

	def verbose(self,message){
		if self.print_verbose{
			self.info(message)
		}
    }
}