import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'ASD Brain Connectivity Predictor',
  description: 'Visualize autism spectrum disorder prediction results with 3D brain connectivity mapping',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="zh-CN">
      <body className="min-h-screen neural-bg">
        <div className="fixed inset-0 bg-gradient-to-br from-slate-900 via-indigo-950/50 to-slate-900 -z-10" />
        {children}
      </body>
    </html>
  )
}
