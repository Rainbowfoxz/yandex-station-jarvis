package main

// Нейросеть в роли Джарвиса — перенос gemini_brain.py без звука: вопрос приходит текстом от Алисы.
// Два вида API: Gemini (REST generateContent) и любой OpenAI-совместимый (/chat/completions).

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"strings"
	"time"
)

const (
	alicePrefix   = "АЛИСА:"
	memoryTime    = 5 * time.Minute // разговор помнится, пока между вопросами меньше 5 минут
	maxTurns      = 10
	modelTimeout  = 12 * time.Second
	trySeconds    = 15 * time.Second // после стольких секунд новые модели не пробуем
	backToPrimary = 10 * time.Minute // удачная запасная модель первая, но через 10 мин снова основная
)

const persona = `Ты — Джарвис, дворецкий-ИИ из «Железного человека», живёшь в умной колонке.
Говоришь по-русски, вежливо и сдержанно, с лёгкой британской иронией, обращаешься «сэр».
Шутки и отсылки — только когда это разрешает подсказка в скобках после вопроса, не больше одной
сухой фразы и не вместо сути ответа. Твой основной юмор — сухая ирония над привычками и
экспериментами самого сэра одной фразой в конце ответа, например: «Физические законы здесь
соблюдаются, по крайней мере, пока вы не пытаетесь снова улучшить тостер» — это образец тона,
каждый раз придумывай новую, про тостер не повторяй. Мистера Старка упоминаешь, только если
подсказка это разрешает. В командах колонке и серьёзных вопросах (здоровье, безопасность)
не шутишь никогда.
Твой ответ читается вслух синтезатором речи, поэтому:
- 1–3 коротких предложения, без списков, markdown, эмодзи и ссылок;
- числа, даты и единицы пиши так, как их произносят.
Колонкой управляет Алиса. Если пользователь просит действие с колонкой или домом — музыку
(включить, остановить, следующий трек, «мою волну»), громкость, таймер, будильник, напоминание,
погоду, время, свет и умный дом, — ответь ровно одной строкой «АЛИСА: <команда для Алисы>»,
например «АЛИСА: включи мою волну» или «АЛИСА: поставь таймер на 10 минут». Больше ничего не пиши.
На всё остальное отвечай сам.`

// Шутку (не больше одной) выбирает жребий: модель сама частоту не соблюдает (см. gemini_brain.py).
var jokes = []struct {
	share float64
	hint  string
}{
	{0.3, "(Можно закончить ответ одной сухой ироничной фразой о привычках сэра. Мистера Старка не упоминай.)"},
	{0.1, "(Можно одну сухую отсылку к мистеру Старку.)"},
}

const noJoke = "(В этом ответе без шуток: ни иронии, ни мистера Старка.)"

func jokeHint() string {
	draw := rand.Float64()
	for _, j := range jokes {
		if draw < j.share {
			return j.hint
		}
		draw -= j.share
	}
	return noJoke
}

type turn struct{ role, text string } // role: user | model

type Brain struct {
	cfg        Config
	client     *http.Client
	history    []turn
	last       time.Time
	model      string // модель, ответившая последней
	fallbackAt time.Time
}

func NewBrain(cfg Config) *Brain {
	return &Brain{cfg: cfg, client: &http.Client{Timeout: modelTimeout}}
}

// Ask -> ("alice", команда) или ("say", текст ответа).
func (b *Brain) Ask(question string) (kind, text string, err error) {
	if time.Since(b.last) > memoryTime {
		b.history = nil
	}
	prompt := question + "\n" + jokeHint()
	answer, err := b.ask(append(b.history, turn{"user", prompt}))
	if err != nil {
		return "", "", err
	}
	b.history = append(b.history, turn{"user", question}, turn{"model", answer})
	if len(b.history) > 2*maxTurns {
		b.history = b.history[len(b.history)-2*maxTurns:]
	}
	b.last = time.Now()
	kind, text = route(answer)
	return kind, text, nil
}

func (b *Brain) ask(turns []turn) (string, error) {
	models := b.cfg.Models
	if b.model != "" && b.model != models[0] && time.Since(b.fallbackAt) > backToPrimary {
		b.model = ""
	}
	order := []string{}
	if b.model != "" {
		order = append(order, b.model)
	}
	for _, m := range models {
		if m != b.model {
			order = append(order, m)
		}
	}
	var lastErr error
	giveUp := time.Now().Add(trySeconds)
	for _, model := range order {
		if lastErr != nil && time.Now().After(giveUp) {
			break
		}
		var text string
		var err error
		if b.cfg.Provider == "gemini" {
			text, err = b.gemini(model, turns)
		} else {
			text, err = b.openai(model, turns)
		}
		if err != nil {
			logf("  %s: %v", model, err)
			lastErr = err
			continue
		}
		if model != models[0] && model != b.model {
			b.fallbackAt = time.Now()
		}
		b.model = model
		return text, nil
	}
	return "", lastErr
}

func (b *Brain) post(url string, headers map[string]string, body any, out any) error {
	data, _ := json.Marshal(body)
	req, err := http.NewRequest("POST", url, bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := b.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode != 200 {
		// В тексте ошибки ключа нет (он в заголовке); обрезаем — нужен только смысл.
		msg := strings.Join(strings.Fields(string(raw)), " ")
		if len(msg) > 200 {
			msg = msg[:200]
		}
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, msg)
	}
	return json.Unmarshal(raw, out)
}

func (b *Brain) gemini(model string, turns []turn) (string, error) {
	type part struct {
		Text    string `json:"text"`
		Thought bool   `json:"thought,omitempty"`
	}
	type content struct {
		Role  string `json:"role,omitempty"`
		Parts []part `json:"parts"`
	}
	contents := []content{}
	for _, t := range turns {
		contents = append(contents, content{t.role, []part{{Text: t.text}}})
	}
	level := "LOW"
	for _, m := range b.cfg.MinimalThinking {
		if m == model {
			level = "MINIMAL"
		}
	}
	body := map[string]any{
		"systemInstruction": content{Parts: []part{{Text: persona}}},
		"contents":          contents,
		"generationConfig": map[string]any{
			"temperature": 0.7, "maxOutputTokens": 400,
			"thinkingConfig": map[string]string{"thinkingLevel": level},
		},
	}
	var out struct {
		Candidates []struct {
			Content content `json:"content"`
		} `json:"candidates"`
	}
	url := "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":generateContent"
	if err := b.post(url, map[string]string{"x-goog-api-key": b.cfg.APIKey}, body, &out); err != nil {
		return "", err
	}
	var sb strings.Builder
	if len(out.Candidates) > 0 {
		for _, p := range out.Candidates[0].Content.Parts {
			if !p.Thought {
				sb.WriteString(p.Text)
			}
		}
	}
	if strings.TrimSpace(sb.String()) == "" {
		return "", errors.New("пустой ответ")
	}
	return sb.String(), nil
}

func (b *Brain) openai(model string, turns []turn) (string, error) {
	type message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	messages := []message{{"system", persona}}
	for _, t := range turns {
		role := t.role
		if role == "model" {
			role = "assistant"
		}
		messages = append(messages, message{role, t.text})
	}
	body := map[string]any{"model": model, "messages": messages, "temperature": 0.7, "max_tokens": 400}
	var out struct {
		Choices []struct {
			Message message `json:"message"`
		} `json:"choices"`
	}
	url := strings.TrimRight(b.cfg.BaseURL, "/") + "/chat/completions"
	if err := b.post(url, map[string]string{"Authorization": "Bearer " + b.cfg.APIKey}, body, &out); err != nil {
		return "", err
	}
	if len(out.Choices) == 0 || strings.TrimSpace(out.Choices[0].Message.Content) == "" {
		return "", errors.New("пустой ответ")
	}
	return out.Choices[0].Message.Content, nil
}

// route: строка «АЛИСА: команда» -> команда Алисе, иначе текст для озвучки.
func route(answer string) (string, string) {
	for _, line := range strings.Split(answer, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(strings.ToUpper(line), alicePrefix) {
			if cmd := strings.Trim(line[len(alicePrefix):], " ."); cmd != "" {
				return "alice", cmd
			}
		}
	}
	return "say", strings.Join(strings.Fields(strings.ReplaceAll(answer, "*", "")), " ")
}
