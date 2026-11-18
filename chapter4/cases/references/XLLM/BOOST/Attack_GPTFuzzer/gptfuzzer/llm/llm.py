class OpenAILLM:
    def predict(self, sequences, question):
        black_list = get_black_list()
        for black in black_list:
            if black in sequences[0]:
                return [0]
        split = question.split()
        cnt = 0
        for word in split:
            if word in sequences[0]:
                cnt += 1
        if cnt/len(split) < 0.5:
            return [0]
        prediction = self.generate(sequences[0])
        for i in range(len(prediction)):
            try:
                prediction[i] = int(prediction[i])
            except:
                prediction[i] = 0
        if prediction[0] == 1:
            print(sequences[0])
        return prediction
