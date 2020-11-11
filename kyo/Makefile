IMG_NAME = mbrl
CONTAINER_NAME = mbrl
CMD = python src/train.py


build:
	docker build -t $(IMG_NAME) .

run:
	docker run -td --name $(CONTAINER_NAME) -e DISPLAY=192.168.99.1:0.0 $(IMG_NAME)
	docker cp src $(CONTAINER_NAME):/root
	docker cp result $(CONTAINER_NAME):/root
	-docker exec -it $(CONTAINER_NAME) $(CMD)
	-docker cp $(CONTAINER_NAME):/root/result .
	docker stop $(CONTAINER_NAME)
	docker rm $(CONTAINER_NAME)

train:
	docker run -td --name $(CONTAINER_NAME) -e DISPLAY=192.168.99.1:0.0 $(IMG_NAME)
	docker cp src $(CONTAINER_NAME):/root
	docker cp result $(CONTAINER_NAME):/root
	-docker exec -it $(CONTAINER_NAME) python src/train.py
	-docker cp $(CONTAINER_NAME):/root/result .
	docker stop $(CONTAINER_NAME)
	docker rm $(CONTAINER_NAME)

evaluate:
	docker run -td --name $(CONTAINER_NAME) -e DISPLAY=192.168.99.1:0.0 $(IMG_NAME)
	docker cp src $(CONTAINER_NAME):/root
	docker cp result $(CONTAINER_NAME):/root
	-docker exec -it $(CONTAINER_NAME) python src/evaluate.py
	-docker cp $(CONTAINER_NAME):/root/result .
	docker stop $(CONTAINER_NAME)
	docker rm $(CONTAINER_NAME)

start:
	docker run -td --name $(CONTAINER_NAME) -e DISPLAY=192.168.99.1:0.0 $(IMG_NAME)
	docker cp src $(CONTAINER_NAME):/root
	docker exec -it $(CONTAINER_NAME) $(CMD)

stop:
	docker stop $(CONTAINER_NAME)
	docker rm $(CONTAINER_NAME)
