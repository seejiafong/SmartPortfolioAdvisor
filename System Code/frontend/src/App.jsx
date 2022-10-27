import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ContextProvider } from './contexts/ContextProvider';
import Dashboard from './Dashboard'

function App() {
    return (
        <div style={{ fontFamily: 'Avenir' }}>
            <BrowserRouter>
                <ContextProvider>
                    <Routes>
                        <Route path='/*' element={(<Dashboard />)} >
                        </Route>                       
                    </Routes>
                </ContextProvider>
            </BrowserRouter>
        </div>
    )
}

export default App
