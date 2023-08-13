package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"

	"github.com/google/uuid"
	"go.uber.org/zap"
)

type OpenAI interface {
	Complete(system string, user string, history []Message, functions []FunctionDefinition) (Message, error)
}

type oaiRequest struct {
	Model     string               `json:"model"`
	Messages  []Message            `json:"messages"`
	Functions []FunctionDefinition `json:"functions,omitempty"`
}

type Message struct {
	Role         string        `json:"role"`
	Content      string        `json:"content,omitempty"`
	FunctionCall *FunctionCall `json:"function_call,omitempty"`
}

type oaiResponse struct {
	Choices []oaiChoice `json:"choices"`
	Error   oaiError    `json:"error"`
}

type oaiChoice struct {
	Message Message `json:"message"`
}

type oaiError struct {
	Message string `json:"message"`
}

type openai struct {
	base  string
	key   string
	model string

	log    *zap.Logger
	client *http.Client
}

type FunctionDefinition struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  Schema `json:"parameters"`
}

type Schema struct {
	Type        string            `json:"type"`
	Description string            `json:"description,omitempty"`
	Properties  map[string]Schema `json:"properties,omitempty"`
	Required    []string          `json:"required,omitempty"`
	Enum        []string          `json:"enum,omitempty"`
	Items       *Schema           `json:"items,omitempty"`
}

type FunctionCall struct {
	Name         string `json:"name"`
	ArgumentsRaw string `json:"arguments"`
}

func (o *openai) Complete(system, user string, history []Message, functions []FunctionDefinition) (Message, error) {
	log := o.log.With(zap.String("requestID", uuid.NewString()), zap.String("model", o.model))
	log.Debug("called completion", zap.String("content", user))

	messages := append([]Message{{Role: "system", Content: system}}, history...)
	messages = append(messages, Message{Role: "user", Content: user})

	request := oaiRequest{
		Model:     o.model,
		Messages:  messages,
		Functions: functions,
	}

	b, err := json.Marshal(request)
	if err != nil {
		log.Error("failed to marshal request", zap.Error(err))
		return Message{}, err
	}
	log.Debug("request data", zap.String("request", string(b)))

	cPath, err := url.JoinPath(o.base, "/v1/chat/completions")
	if err != nil {
		log.Error("failed to create url for chat completion", zap.Error(err))
		return Message{}, fmt.Errorf("failed to create url for chat completion")
	}

	req, err := http.NewRequest("POST", cPath, bytes.NewReader(b))
	if err != nil {
		log.Error("failed to create OpenAI request", zap.Error(err))
		return Message{}, err
	}
	req.Header.Add("Content-Type", "application/json")

	if o.key != "" {
		req.Header.Add("Authorization", fmt.Sprintf("Bearer %s", o.key))
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		log.Error("failed to call OpenAI service", zap.Error(err))
		return Message{}, err
	}
	defer resp.Body.Close()

	b, err = io.ReadAll(resp.Body)
	if err != nil {
		log.Error("failed to read response body", zap.Error(err))
		return Message{}, err
	}

	log.Debug("OpenAI response", zap.String("content", string(b)))

	var response oaiResponse
	err = json.Unmarshal(b, &response)
	if err != nil {
		log.Error("failed to unmarshal OpenAI response", zap.Error(err))
		return Message{}, err
	}

	if resp.StatusCode != 200 {
		err = fmt.Errorf(response.Error.Message)
		log.Error("response status is not success", zap.Error(err))
		return Message{}, err
	}

	if len(response.Choices) != 1 {
		err = fmt.Errorf("unexpected number of choices in response")
		log.Error("unexpected number of choices in response", zap.Error(err))
		return Message{}, err
	}

	msg := response.Choices[0].Message
	log.Debug("request completed successfully", zap.Any("result", msg))

	return msg, nil
}

func New(log *zap.Logger) (OpenAI, error) {
	key := os.Getenv("OPENAI_API_KEY")
	base := os.Getenv("OPENAI_API_BASE")
	model := os.Getenv("OPENAI_API_MODEL")
	var openaibase bool
	if base == "" {
		base = "https://api.openai.com"
		openaibase = true
	}

	if model == "" {
		model = "gpt-3.5-turbo-0613"
	}

	if key == "" && openaibase {
		return nil, fmt.Errorf("OPENAI_API_KEY must be supplied if using openai service")
	}

	if log == nil {
		log = zap.NewNop()
	}

	log = log.Named("OpenAI")

	return &openai{
		log:    log,
		base:   base,
		key:    key,
		model:  model,
		client: http.DefaultClient,
	}, nil
}
