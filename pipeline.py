import luigi
from tasks.Preprocessing.preprocessing_task import ProcessTrainAndTestData
from tasks.Feature_Engineering.feature_engineering_task import FeatureEngineering
from tasks.Train.train_task import Train



def create_task_list():
    tasks = []
    tasks.append(ProcessTrainAndTestData())
    tasks.append(FeatureEngineering())
    tasks.append(Train())

    return tasks

def main():
    tasks = create_task_list()
    luigi.build(
        tasks,
        local_scheduler=True,
    )

    #success = all([task.complete() for task in tasks])
    #return success




if __name__ == '__main__':
    main()