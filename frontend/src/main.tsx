import React from 'react'
import ReactDOM from 'react-dom/client'
import { SWRConfig } from 'swr'
import Account from './Account'
import { api } from './api'
import { RuntimeBoundary } from './components'
import './styles.css'

ReactDOM.createRoot(document.getElementById('root')!).render(<React.StrictMode><RuntimeBoundary><SWRConfig value={{ fetcher: api, revalidateOnFocus: false, shouldRetryOnError: false }}><Account/></SWRConfig></RuntimeBoundary></React.StrictMode>)
