all: *.java
	mkdir -p build
	javac -d build *.java
	cd build; jar cvfe tune.jar ZMERT *; cd ..

clean:
	rm -rf build
