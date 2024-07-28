# prefect
from prefect import flow, task

# 
@task
def testing_task():
    return 'hello world'

@flow
def testing_flow():
    message = testing_task()
    print(message)

if __name__ == '__main__':
    testing_flow()