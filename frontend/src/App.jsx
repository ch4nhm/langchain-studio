import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { v4 as uuidv4 } from 'uuid'
import { Copy, RefreshCw, Info } from 'lucide-react'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState(uuidv4())
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async (e) => {
    e?.preventDefault()
    if (!input.trim()) return

    const userMsg = { role: 'user', content: input }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setIsLoading(true)

    const aiMsgIndex = messages.length + 1
    setMessages(prev => [...prev, { role: 'assistant', content: '', sources: [] }])

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: input, session_id: sessionId })
      })

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let done = false

      while (!done) {
        const { value, done: readerDone } = await reader.read()
        done = readerDone
        if (value) {
          const chunkStr = decoder.decode(value)
          const lines = chunkStr.split('\n')
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6)
              if (data === '[DONE]') {
                setIsLoading(false)
                break
              }
              try {
                const parsed = JSON.parse(data)
                if (parsed.chunk) {
                  setMessages(prev => {
                    const newMessages = [...prev]
                    newMessages[aiMsgIndex].content += parsed.chunk
                    return newMessages
                  })
                }
              } catch (e) {
                console.error("Parse error", e)
              }
            }
          }
        }
      }
    } catch (error) {
      console.error(error)
      setIsLoading(false)
    }
  }

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text)
  }

  return (
    <div className="flex flex-col h-full bg-gray-50">
      <header className="bg-white shadow px-6 py-4 flex items-center justify-between">
        <h1 className="text-xl font-bold text-gray-800">LangChain AI Assistant</h1>
        <span className="text-sm text-gray-500">Session: {sessionId.substring(0,8)}</span>
      </header>
      
      <main className="flex-1 overflow-y-auto p-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[80%] rounded-lg p-4 shadow ${msg.role === 'user' ? 'bg-blue-600 text-white' : 'bg-white text-gray-800'}`}>
                {msg.role === 'assistant' ? (
                  <div className="prose prose-sm max-w-none">
                    <ReactMarkdown 
                      remarkPlugins={[remarkGfm]}
                      components={{
                        code({node, inline, className, children, ...props}) {
                          const match = /language-(\w+)/.exec(className || '')
                          return !inline && match ? (
                            <div className="relative group">
                              <button onClick={() => copyToClipboard(String(children))} className="absolute right-2 top-2 text-gray-400 hover:text-white">
                                <Copy size={16} />
                              </button>
                              <SyntaxHighlighter language={match[1]} PreTag="div" {...props}>
                                {String(children).replace(/\n$/, '')}
                              </SyntaxHighlighter>
                            </div>
                          ) : (
                            <code className={className} {...props}>{children}</code>
                          )
                        }
                      }}
                    >
                      {msg.content || (isLoading && idx === messages.length - 1 ? '...' : '')}
                    </ReactMarkdown>
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="mt-4 pt-4 border-t border-gray-200">
                        <button className="flex items-center text-sm text-gray-500 hover:text-blue-600">
                          <Info size={14} className="mr-1" /> View Sources
                        </button>
                      </div>
                    )}
                  </div>
                ) : (
                  <div>{msg.content}</div>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </main>

      <footer className="bg-white border-t p-4">
        <div className="max-w-4xl mx-auto">
          <form onSubmit={handleSend} className="flex gap-4">
            <input 
              type="text" 
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..." 
              className="flex-1 rounded-lg border border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={isLoading}
            />
            <button 
              type="submit" 
              disabled={isLoading || !input.trim()}
              className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              Send
            </button>
          </form>
        </div>
      </footer>
    </div>
  )
}

export default App
