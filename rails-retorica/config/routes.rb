# -*- encoding : utf-8 -*-
Retorica::Application.routes.draw do

  root :to => redirect('dashboards/first')

  get '/about' => 'welcome#about'

  resources :dashboards, :except => [:new, :create, :edit, :update, :destroy] do
    collection { get :first }
  end

  resources :deputados, :except => [:new, :create, :update, :destroy]

end
