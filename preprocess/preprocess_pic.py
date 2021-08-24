import math
import sys
import cPickle as pickle
from datetime import datetime
import random
import argparse
import entropy_analysis

global ARGS

def get_ICD10s_from_pic_file(fileName, hadmToMap):
  picFile = open(fileName, 'r')  # row_id,subject_id,hadm_id,seq_num,ICD10_code_CN
  picFile.readline()
  number_of_null_ICD10_codes = 0
  for line in picFile:			 #   0  ,     1    ,    2   ,   3  ,    4
    tokens = line.strip().split(',')
    hadm_id = int(tokens[2])
    if (len(tokens[4]) == 0):  # ignore diagnoses where ICD10_code_CN is null
      number_of_null_ICD10_codes += 1
      continue;

    # Remove alphanumeric characters
    tokens[4] = tokens[4].replace('/','')
    tokens[4] = tokens[4].replace('.', '')
    tokens[4] = tokens[4].replace('+', '')
    ICD10_code = tokens[4]

    if hadm_id in hadmToMap:
      hadmToMap[hadm_id].add(ICD10_code)
    else:
      hadmToMap[hadm_id] = set()              #use set to avoid repetitions
      hadmToMap[hadm_id].add(ICD10_code)
  for hadm_id in hadmToMap.keys():
    hadmToMap[hadm_id] = list(hadmToMap[hadm_id])   #convert to list, as the rest of the codes expects
  picFile.close()
  print '-Number of null ICD10 codes in file ' + fileName + ': ' + str(number_of_null_ICD10_codes)


def convert_department_to_float(department):
  #very specific to PIC ADMISSIONS.csv
  code = 0
  if department == 'General ICU': code = 0
  elif department == 'Nine ward': code = 1
  elif department == 'Hematology department(1)': code = 2
  elif department == 'General surgery department': code = 3
  elif department == 'Urinary surgery department(1)': code = 4
  elif department == 'Rheumatology department': code = 5
  elif department == 'Hematology department(2)': code = 6
  elif department == 'Urinary surgery department(2)': code = 7
  elif department == 'Nephrology department(2)': code = 8
  elif department == 'Nephrology department(1)': code = 9
  elif department == 'Pediatric internal medicine(1)': code = 10
  elif department == 'Orthopedics/Neurology department': code = 11
  elif department == 'Hematology department(3)': code = 12
  elif department == 'PICU': code = 13
  elif department == 'NICU': code = 14
  elif department == 'SICU': code = 15
  elif department == 'Orthopedics/Traumatology department': code = 16
  elif department == 'CICU': code = 17
  elif department == 'Thoracic surgery/Oncology department': code = 18
  elif department == 'General surgery/Neonatology surgery department': code = 19
  elif department == 'Cardiac surgery department': code = 20
  elif department == 'General surgery/Endoscopy department': code = 21
  elif department == 'Burn/Neurosurgery department': code = 22
  elif department == 'Gastroenterology department': code = 23
  elif department == 'Neonatology department(1)': code = 24
  elif department == 'Respiratory medicine department(1)': code = 25
  elif department == 'Infectious diseases department': code = 26
  elif department == 'Cardiovascular department': code = 27
  elif department == 'Endocrinology department': code = 28
  elif department == 'Neurology department': code = 29
  elif department == 'Neonatology department(2)': code = 30
  elif department == 'Ophthalmology department/ENT': code = 31
  elif department == 'Respiratory medicine department(2)': code = 32
  else: print 'ERROR in admission department value'
  return code

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('admissions_file', type=str, default='ADMISSIONS.csv', help='The ADMISSIONS.csv file of PIC distribution.')
  parser.add_argument('diagnoses_file', type=str, default='', help='The DIAGNOSES_ICD.csv file of PIC distribution.')
  parser.add_argument('output_prefix', type=str, default='preprocessing', help='The output file radical name.')
  parser.add_argument('--data_partition', type=str, default='[90,10]', help='Provide an array with two values that sum up 100.')
  argsTemp = parser.parse_args()
  return argsTemp

if __name__ == '__main__':
  ARGS = parse_arguments()
  partitions = [int(strDim) for strDim in ARGS.data_partition[1:-1].split(',')]
  Ordered_internalCodesMap = {}

  #one line of the admissions file contains one admission hadm_id of one subject_id at a given time admittime
  print 'Building Maps: hadm_id to admtttime, duration, and department; and Map: subject_id to set of all its hadm_ids'
  subjectTOhadms_Map = {}
  hadmTOadmttime_Map = {}					   					#   0  ,     1    ,    2  ,     3   ,    4
  hadmTOduration_Map = {}
  hadmTOinterval_Map = {}
  hadmTOadmDep_Map = {}
  pic_ADMISSIONS_csv = open(ARGS.admissions_file, 'r')
  # row_id,subject_id,hadm_id,admittime,dischtime,deathtime,admission_department,discharge_department,insurance,language,religion,marital_status,ethnicity,
  pic_ADMISSIONS_csv.readline()

  initial_number_of_admissions = 0
  previous_subject = 0
  previous_admission = 0

  for line in pic_ADMISSIONS_csv:
    initial_number_of_admissions += 1
    tokens = line.strip().split(',')
    subject_id = int(tokens[1])
    hadm_id = int(tokens[2])
    admittime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
    dischargetime = datetime.strptime(tokens[4], '%Y-%m-%d %H:%M:%S')
    admissionDepartment = tokens[6]

    # hadmTOadmttime_Map(hadm_id) -> duration of admission in hours
    temp = dischargetime - admittime
    hadmTOduration_Map[hadm_id] = temp.days * 24 + temp.seconds / 3600  # duration in hours

    #keep track of the admission amount of time
    #hadmTOadmttime_Map(hadm_id) -> time of admission hadm_id
    hadmTOadmttime_Map[hadm_id] = admittime

    #on a subject basis
    if subject_id == previous_subject:
      # keep track of the time since the last admission in days
      temp = admittime - hadmTOadmttime_Map[previous_admission]
      hadmTOinterval_Map[hadm_id] = temp.days + temp.seconds / 3600 / 24  # time since the last admission in days
    else:
      hadmTOinterval_Map[hadm_id] = 0  # 1st interval since the last admission is 0

    previous_admission = hadm_id
    previous_subject = subject_id

    #register department of admission
    hadmTOadmDep_Map[hadm_id] = [convert_department_to_float(admissionDepartment)]

    #subjectTOhadms_Map(subject_id) -> set of hadms for subject_id
    if subject_id in subjectTOhadms_Map: subjectTOhadms_Map[subject_id].append(hadm_id)
    else: subjectTOhadms_Map[subject_id] = [hadm_id] #the brackets indicate that it will be a list
  pic_ADMISSIONS_csv.close()
  print '-Initial number of admissions: ' + str(initial_number_of_admissions)
  print '-Initial number of subjects: ' + str(len(subjectTOhadms_Map))
  hadmToICD10CODEs_Map = {}
  
  if len(ARGS.diagnoses_file) > 0:
    #one line in the diagnoses file contains only one diagnose code (ICD10) for one admission hadm_id
    print 'Building Map: hadm_id to set of ICD10 codes from DIAGNOSES_ICD'
    get_ICD10s_from_pic_file(ARGS.diagnoses_file, hadmToICD10CODEs_Map)

  print '-Number of valid admissions (at least one diagnosis): ' + str(len(hadmToICD10CODEs_Map))

  #Cleaning up inconsistencies
  #some tuples in the diagnoses table have ICD10 empty; we clear the admissions without diagnoses from all the maps
  #this may cause the presence of patients (subject_ids) with 0 admissions hadm_id; we clear these guys too
  #We also clean admissions in which admission time < discharge time
  number_of_admissions_without_diagnosis = 0
  number_of_subjects_without_valid_admissions = 0
  print 'Cleaning up admissions without diagnoses'
  for subject_id, hadmList in subjectTOhadms_Map.items():   #hadmTOadmttime_Map,subjectTOhadms_Map,hadm_cid10s_Map
    hadmListCopy = list(hadmList)    #copy the list, iterate over the copy, edit the original; otherwise, iteration problems
    for hadm_id in hadmListCopy:
      if hadm_id not in hadmToICD10CODEs_Map.keys():  #map hadmToICD10CODEs_Map is already valid by creation
        number_of_admissions_without_diagnosis += 1
        del hadmTOadmttime_Map[hadm_id]     #delete by key
        del hadmTOduration_Map[hadm_id]
        del hadmTOinterval_Map[hadm_id]
        del hadmTOadmDep_Map[hadm_id]
        hadmList.remove(hadm_id)
    if len(hadmList) == 0:					      #toss off subject_id without admissions
      number_of_subjects_without_valid_admissions += 1
      del subjectTOhadms_Map[subject_id]     #delete by value
  print '-Number of admissions without diagnosis: ' + str(number_of_admissions_without_diagnosis)
  print '-Number of admissions after cleaning: ' + str(len(hadmToICD10CODEs_Map))
  print '-Number of subjects without admissions: ' + str(number_of_subjects_without_valid_admissions)
  print '-Number of subjects after cleaning: ' + str(len(subjectTOhadms_Map))

  #since the data in the database is not necessarily time-ordered
  #here we sort the admissions (hadm_id) according to the admission time (admittime)
  #after this, we have a list subjectTOorderedHADM_IDS_Map(subject_id) -> admission-time-ordered set of ICD10 codes
  print 'Building Map: subject_id to admission-ordered (admittime, ICD10s set) and cleaning one-admission-only patients'
  subjectTOorderedHADM_IDS_Map = {}
	#for each admission hadm_id of each patient subject_id
  number_of_subjects_with_only_one_visit = 0
  for subject_id, hadmList in subjectTOhadms_Map.iteritems():
    if len(hadmList) < 2:
      number_of_subjects_with_only_one_visit += 1
      continue  #discard subjects with only 2 admissions
    #sorts the hadm_ids according to date admttime
    #only for the hadm_id in the list hadmList
    sortedList = sorted([(hadmTOadmttime_Map[hadm_id], hadmToICD10CODEs_Map[hadm_id], hadm_id) for hadm_id in hadmList])
    # each element in subjectTOorderedHADM_IDS_Map is a key-value (subject_id, (admittime, ICD10_List, hadm_id))
    subjectTOorderedHADM_IDS_Map[subject_id] = sortedList
  print '-Number of discarded subjects with only one admission: ' + str(number_of_subjects_with_only_one_visit)
  print '-Number of subjects after ordering: ' + str(len(subjectTOorderedHADM_IDS_Map))

  print 'Converting maps to lists in preparation for dump'
  all_subjectsListOfCODEsList_LIST = []
  #for each subject_id, get its key-value (subject_id, (admittime, CODESs_List))
  for subject_id, time_ordered_CODESs_List in subjectTOorderedHADM_IDS_Map.iteritems():
    subject_list_of_CODEs_List = []
    #for each admission (admittime, CODESs_List) build lists of time and CODEs list
    for admission in time_ordered_CODESs_List:   		#each element in time_ordered_CODESs_List is a tripple (admittime, ICD10_List, hadm_id)
	    #here, admission = [admittime, ICD10_List, hadm_id)
      subject_list_of_CODEs_List.append((admission[1],admission[2]))  #build list of lists of the admissions' CODEs of the current subject_id, stores hadm_id together
    #lists of lists, one entry per subject_id
    all_subjectsListOfCODEsList_LIST.append(subject_list_of_CODEs_List)	#build list of list of lists of the admissions' ICD10s - one entry per subject_id

  CODES_distributionMAP = entropy_analysis.writeDistributions(ARGS.admissions_file, hadmToICD10CODEs_Map, subjectTOhadms_Map, all_subjectsListOfCODEsList_LIST)
  for i, key in enumerate(CODES_distributionMAP):
    Ordered_internalCodesMap[key[0]] = i 
  entropy_analysis.computeShannonEntropyDistribution(all_subjectsListOfCODEsList_LIST, CODES_distributionMAP, ARGS.admissions_file)
	
  #Randomize the order of the patients at the first dimension
  random.shuffle(all_subjectsListOfCODEsList_LIST)

  duration_of_admissionsListOfLists = []  #list of lists of duration of admissions, one list for each patient (subjet_id)
  interval_since_last_admissionListOfLists = []
  dep_of_admissionsListOfLists = []
  new_all_subjectsListOfCODEsList_LIST = []
  final_number_of_admissions = 0
  #Here we convert the database codes to internal sequential codes
  #we use the same for to build lists of interval, duration and department
  print 'Converting database ids to sequential integer ids'
  for subject_list_of_CODEs_List in all_subjectsListOfCODEsList_LIST:
    new_subject_list_of_CODEs_List = []
    duration_of_admissionsList = []
    interval_since_last_admissionList = []
    dep_of_admissionsList = []
    for CODEs_List in subject_list_of_CODEs_List:
      final_number_of_admissions += 1
      new_CODEs_List = []
      hadm_id = CODEs_List[1]
      durationTemp = hadmTOduration_Map[hadm_id]
      intervalTemp = hadmTOinterval_Map[hadm_id]
      #we bypass admissions with 0 or negative durations
      if durationTemp <= 0 or intervalTemp < 0:
        continue

      duration_of_admissionsList.append(durationTemp)
      interval_since_last_admissionList.append(intervalTemp)
      dep_of_admissionsList.append(hadmTOadmDep_Map[hadm_id])

      for CODE in CODEs_List[0]:
        new_CODEs_List.append(Ordered_internalCodesMap[CODE])   #newVisit is the CODEs_List, but with the new sequential ids
      new_subject_list_of_CODEs_List.append(new_CODEs_List)		#new_subject_list_of_CODEs_List is the subject_list_of_CODEs_List, but with the id given by its frequency

    #when we bypass admissions with 0 or negative durations, we might create patients with only one admission, which we also bypass
    if len(new_subject_list_of_CODEs_List) > 1:
      duration_of_admissionsListOfLists.append(duration_of_admissionsList)
      interval_since_last_admissionListOfLists.append(interval_since_last_admissionList)
      dep_of_admissionsListOfLists.append(dep_of_admissionsList)
      new_all_subjectsListOfCODEsList_LIST.append(new_subject_list_of_CODEs_List)	#new_all_subjectsListOfCODEsList_LIST is the all_subjectsListOfCODEsList_LIST, but with the new sequential ids

  print ''
  nCodes = len(Ordered_internalCodesMap)
  print '-Number of actually used DIAGNOSES codes: '+ str(nCodes)

  print '-Final number of subjects: ' + str(len(new_all_subjectsListOfCODEsList_LIST))
  print '-Final number of admissions: ' + str(final_number_of_admissions)

  #Partitioning data
  if (len(partitions) >= 1):
    total_patients_dumped = 0
    print 'Writing ' + str(partitions[0]) + '% of the patients read from file ' + ARGS.admissions_file
    index_of_last_patient_to_dump = int(math.ceil(len(new_all_subjectsListOfCODEsList_LIST)*int(partitions[0])/100))
    pickle.dump(new_all_subjectsListOfCODEsList_LIST[0:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.train', 'wb'), -1)
    pickle.dump(duration_of_admissionsListOfLists[0:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.DURATION.train', 'wb'), -1)
    pickle.dump(interval_since_last_admissionListOfLists[0:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.INTERVAL.train', 'wb'), -1)
    pickle.dump(dep_of_admissionsListOfLists[0:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.DEP.train', 'wb'), -1)
    print '   Patients from 0 to ' + str(index_of_last_patient_to_dump)
    print '   Success, file: ' + ARGS.output_prefix + '_' + str(nCodes) + '.train created'
    total_patients_dumped += index_of_last_patient_to_dump

    if (len(partitions) >= 2):
      print 'Writing ' + str(partitions[1]) + '% of the patients read from file ' + ARGS.admissions_file
      index_of_first_patient_to_dump = index_of_last_patient_to_dump
      index_of_last_patient_to_dump = index_of_first_patient_to_dump + int(math.ceil(len(new_all_subjectsListOfCODEsList_LIST)*int(partitions[1])/100))
      pickle.dump(new_all_subjectsListOfCODEsList_LIST[index_of_first_patient_to_dump:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.test', 'wb'), -1)
      pickle.dump(duration_of_admissionsListOfLists[index_of_first_patient_to_dump:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.DURATION.test', 'wb'), -1)
      pickle.dump(interval_since_last_admissionListOfLists[index_of_first_patient_to_dump:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.INTERVAL.test', 'wb'), -1)
      pickle.dump(dep_of_admissionsListOfLists[index_of_first_patient_to_dump:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.DEP.test', 'wb'), -1)
      print '   Patients from ' + str(index_of_first_patient_to_dump) + ' to ' + str(index_of_last_patient_to_dump)
      print '   Success, file: ' + ARGS.output_prefix + '_' + str(nCodes) + '.test created'
      total_patients_dumped += index_of_last_patient_to_dump - index_of_first_patient_to_dump

      if (len(partitions) >= 3):
        print 'Writing ' + str(partitions[2]) + '% of the patients read from file ' + ARGS.admissions_file
        index_of_first_patient_to_dump = index_of_last_patient_to_dump
        pickle.dump(new_all_subjectsListOfCODEsList_LIST[index_of_first_patient_to_dump:],open(ARGS.output_prefix + '_' + str(nCodes) + '.valid', 'wb'), -1)
        pickle.dump(duration_of_admissionsListOfLists[index_of_first_patient_to_dump:], open(ARGS.output_prefix + '_' + str(nCodes) + '.DURATION.valid', 'wb'), -1)
        pickle.dump(interval_since_last_admissionListOfLists[index_of_first_patient_to_dump:], open(ARGS.output_prefix + '_' + str(nCodes) + '.INTERVAL.valid', 'wb'), -1)
        pickle.dump(dep_of_admissionsListOfLists[index_of_first_patient_to_dump:], open(ARGS.output_prefix + '_' + str(nCodes) + '.DEP.valid', 'wb'), -1)
        print '   Patients from ' + str(index_of_first_patient_to_dump) + ' to the end of the file'
        print '   Success, file: ' + ARGS.output_prefix + '_' + str(nCodes) + '.valid created'
        total_patients_dumped += len(new_all_subjectsListOfCODEsList_LIST) - total_patients_dumped
        print 'Total of dumped patients: ' + str(total_patients_dumped) + ' out of ' + str(len(new_all_subjectsListOfCODEsList_LIST))
  else:
    print 'Error, please provide data partition scheme. E.g, [80,10,10], for 80\% train, 10\% test, and 10\% validation.'
