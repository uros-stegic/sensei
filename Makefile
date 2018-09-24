BUILD_FOLDER	= build
PROG_NAME		= libsensei.so
TEST_FOLDER		= test
PROG			= $(BUILD_FOLDER)/$(PROG_NAME)

.PHONY: clean $(BUILD_FOLDER)

$(PROG): $(BUILD_FOLDER)
	@cd $<; \
	cmake ..; \
	make -j$(nproc);
	@cd $(TEST_FOLDER); \
	make; \

$(BUILD_FOLDER):
	@mkdir -p $@

clean:
	@rm -rf $(BUILD_FOLDER)
	@cd $(TEST_FOLDER); \
	make clean;

debug: $(PROG)
	@cd $(BUILD_FOLDER); \
	exec gdb --args ./$(PROG_NAME)

run: $(PROG)
	@echo " "
	@echo "|_/^\_/^\_/^\_/^\_/^\_/^\_/^\_|  Executing program  |_/^\_/^\_/^\_/^\_/^\_/^\_/^\_|"
	@echo " "
	@cd $(TEST_FOLDER); \
	time ./test
	@echo " "
	@echo "|_/^\_/^\_/^\_/^\_/^\_/^\_/^\_/^\_/^\_/^\_/^\_/^\_/^\_/^\_/^\_/^\_/^\_/^\_/^\_/^\_|"
	@echo " "

